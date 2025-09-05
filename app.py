from io import StringIO
import os
import requests
import gradio as gr
import pandas as pd
from datetime import datetime
import plotly.express as px
from gpt4all import GPT4All

# ---------------- Chatbot Class ----------------
class PersonalFinanceChatbot:
    def __init__(self):
        # Model path and download URL
        self.model_path = "orca-mini-3b-gguf2-q4_0.gguf"
        self.model_url = "https://huggingface.co/your-username/your-model/resolve/main/orca-mini-3b-gguf2-q4_0.gguf"

        # Download model if not exists
        if not os.path.exists(self.model_path):
            print("Downloading GPT4All model...")
            response = requests.get(self.model_url, stream=True)
            with open(self.model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Model downloaded.")

        # Load GPT4All model
        self.model = GPT4All(self.model_path)
        print(f"✅ GPT4All model loaded: {self.model_path}")

        # Store user profiles
        self.user_profiles = {}

    # ---------------- Methods ----------------
    def create_user_profile(self, name, age, user_type, income, expenses, goals):
        profile = {
            'name': name,
            'age': age,
            'user_type': user_type,
            'income': income,
            'monthly_expenses': expenses,
            'goals': goals,
            'created_at': datetime.now().isoformat()
        }
        self.user_profiles[name] = profile
        return f"Profile created for {name}! Personalized guidance is ready."

    def generate_response(self, prompt, user_type, max_tokens=150):
        system_prompt = self._get_system_prompt(user_type)
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nFinancial Assistant:"

        try:
            # Correct GPT4All usage with session
            with self.model.chat_session() as session:
                reply = session.generate(full_prompt, max_tokens=max_tokens)
            return reply
        except Exception as e:
            print(f"⚠️ Error generating response: {e}")
            return "Sorry, I couldn't generate a response. Please try again."

    def _get_system_prompt(self, user_type):
        if user_type == "Student":
            return ("You are a friendly financial advisor speaking to a college student. "
                    "Use simple language, focus on budgeting basics, student loans, part-time income, and building good habits.")
        else:
            return ("You are a professional financial advisor. Provide detailed advice on "
                    "investments, tax strategies, retirement planning, and wealth building.")

    def generate_budget_summary(self, username):
        if username not in self.user_profiles:
            return "Please create a user profile first!", None

        profile = self.user_profiles[username]
        income = profile['income']
        expenses = profile['monthly_expenses']
        disposable_income = income - expenses

        budget_data = {
            'Category': ['Housing', 'Food', 'Transportation', 'Entertainment', 'Savings', 'Other'],
            'Recommended %': [30, 15, 15, 10, 20, 10],
            'Recommended Amount': [income * 0.30, income * 0.15, income * 0.15,
                                   income * 0.10, income * 0.20, income * 0.10],
            'Current Amount': [expenses * 0.4, expenses * 0.25, expenses * 0.2,
                               expenses * 0.1, disposable_income * 0.8, expenses * 0.05]
        }
        df = pd.DataFrame(budget_data)
        fig = px.bar(df, x='Category', y=['Recommended Amount', 'Current Amount'],
                     title=f"Budget Analysis for {username}", barmode='group')

        savings_rate = (disposable_income / income) * 100 if income > 0 else 0
        user_type = profile['user_type']

        if user_type == "Student":
            summary = (f"🎓 Student Budget Summary for {username}\n\n"
                       f"Monthly Income: ${income:,.2f}\n"
                       f"Monthly Expenses: ${expenses:,.2f}\n"
                       f"Disposable Income: ${disposable_income:,.2f}\n"
                       f"Savings Rate: {savings_rate:.1f}%\n\n"
                       f"- Focus on a $500 emergency fund first\n"
                       f"- Use student discounts and free resources\n"
                       f"- Even small savings add up!")
        else:
            summary = (f"💼 Professional Budget Summary for {username}\n\n"
                       f"Monthly Income: ${income:,.2f}\n"
                       f"Monthly Expenses: ${expenses:,.2f}\n"
                       f"Disposable Income: ${disposable_income:,.2f}\n"
                       f"Savings Rate: {savings_rate:.1f}%\n\n"
                       f"- Target savings rate: 20%+\n"
                       f"- Emergency fund goal: ${expenses * 6:,.2f}\n"
                       f"- Increase retirement contributions")

        return summary, fig

    def analyze_spending_patterns(self, username, spending_data):
        if username not in self.user_profiles:
            return "Please create a user profile first!", None

        try:
            data = pd.read_csv(StringIO(spending_data), header=None, names=['Category', 'Amount'])
            if data.empty:
                return "No spending data provided.", None
        except Exception as e:
            return f"⚠️ Invalid spending data format. Use CSV like:\nFood,500\nClothes,200\nTransport,150\nError: {e}", None

        data['Amount'] = pd.to_numeric(data['Amount'], errors='coerce').fillna(0)
        fig = px.pie(data, values='Amount', names='Category',
                     title=f"Spending Pattern Analysis for {username}")
        total_spending = data['Amount'].sum()
        top_category = data.loc[data['Amount'].idxmax(), 'Category']

        insights = (f"📊 Spending Analysis\n\n"
                    f"Total Spending: ${total_spending:,.2f}\n"
                    f"Highest Category: {top_category}\n\n"
                    f"- Review your top spending areas\n"
                    f"- Set category-wise limits\n"
                    f"- Look for optimization opportunities")
        return insights, fig

    def get_financial_tips(self, user_type, topic):
        tips = {
            "Student": {
                "Budgeting": ["Use the 50/30/20 rule", "Track every expense for a month", "Use student discounts"],
                "Investing": ["Start with index funds", "Learn about compound interest", "Invest small but regularly"],
            },
            "Professional": {
                "Budgeting": ["Automate savings", "Zero-based budgeting", "Track net worth monthly"],
                "Investing": ["Diversify portfolio", "Max out retirement accounts", "Rebalance quarterly"],
            }
        }
        category = topic.title()
        user_tips = tips.get(user_type, {}).get(category, ["Stay consistent with your finances!"])
        return f"💡 {user_type} Tips - {category}:\n" + "\n".join([f"{i+1}. {t}" for i, t in enumerate(user_tips)])


# ---------------- Gradio Interface ----------------
def create_interface():
    chatbot = PersonalFinanceChatbot()
    with gr.Blocks(title="Personal Finance Chatbot", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🏦 Personal Finance Chatbot\n### Get personalized financial advice!")

        with gr.Tabs():
            with gr.TabItem("💬 Financial Chat"):
                gr.ChatInterface(
                    fn=lambda message, history, user_type: chatbot.generate_response(message, user_type),
                    additional_inputs=[gr.Dropdown(choices=["Student", "Professional"], value="Student", label="User Type")],
                    title="Ask your financial questions here!",
                    description="Get personalized advice based on your profile type."
                )

            with gr.TabItem("👤 Profile Setup"):
                name = gr.Textbox(label="Name")
                age = gr.Number(label="Age", value=25)
                user_type = gr.Dropdown(choices=["Student", "Professional"], label="User Type", value="Student")
                income = gr.Number(label="Monthly Income ($)", value=3000)
                expenses = gr.Number(label="Monthly Expenses ($)", value=2000)
                goals = gr.Textbox(label="Financial Goals")
                create_btn = gr.Button("Create Profile")
                profile_out = gr.Textbox(label="Profile Status")
                create_btn.click(fn=chatbot.create_user_profile,
                                 inputs=[name, age, user_type, income, expenses, goals],
                                 outputs=profile_out)

            with gr.TabItem("📊 Budget Analysis"):
                username = gr.Textbox(label="Username")
                btn = gr.Button("Generate Budget Summary")
                summary = gr.Textbox(label="Budget Summary", lines=10)
                chart = gr.Plot(label="Budget Visualization")
                btn.click(fn=chatbot.generate_budget_summary, inputs=username, outputs=[summary, chart])

            with gr.TabItem("🔍 Spending Insights"):
                uname = gr.Textbox(label="Username")
                spend_data = gr.Textbox(
                    label="Spending Data (CSV)",
                    lines=5,
                    placeholder="Example:\nFood,500\nClothes,200\nTransport,150"
                )
                analyze_btn = gr.Button("Analyze Spending")
                insights = gr.Textbox(label="Spending Insights", lines=10)
                chart2 = gr.Plot(label="Spending Visualization")
                analyze_btn.click(fn=chatbot.analyze_spending_patterns, inputs=[uname, spend_data], outputs=[insights, chart2])

            with gr.TabItem("💡 Financial Tips"):
                utype = gr.Dropdown(choices=["Student", "Professional"], label="User Type", value="Student")
                topic = gr.Dropdown(choices=["Budgeting", "Investing"], label="Topic", value="Budgeting")
                tips_btn = gr.Button("Get Tips")
                tips_out = gr.Textbox(label="Financial Tips", lines=10)
                tips_btn.click(fn=chatbot.get_financial_tips, inputs=[utype, topic], outputs=tips_out)

        gr.Markdown("---\n**Disclaimer:** This chatbot provides general financial information only.")

    return interface


# ---------------- Launch ----------------
if __name__ == "__main__":
    app = create_interface()
    # Local launch
    app.launch(server_name="127.0.0.1", server_port=7860, debug=True)
    # For public link, uncomment:
    # app.launch(share=True)
