from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5-20251001")

SYSTEM_PROMPT = """You are an expert AI Accountant with deep knowledge in:
- Bookkeeping and financial record-keeping
- Tax planning and preparation (income tax, GST/VAT, corporate tax)
- Financial statement analysis (balance sheets, income statements, cash flow)
- Budgeting and forecasting
- Accounts payable and receivable management
- Payroll processing
- Audit preparation and compliance
- Cost accounting and profitability analysis
- Invoice and expense management
- Financial regulations and standards (GAAP, IFRS)

Always provide accurate, professional financial guidance. When dealing with tax or legal matters,
remind the user to consult a licensed CPA or tax professional for their specific jurisdiction.
Format numbers clearly, use proper accounting terminology, and show calculations step by step when needed."""

system = SystemMessage(content=SYSTEM_PROMPT)
history = [system]


def chat(user_input: str) -> str:
    history.append(HumanMessage(content=user_input))
    response = model.invoke(history)
    history.append(AIMessage(content=response.content))
    return response.content


def main():
    print("AI Accountant ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        reply = chat(user_input)
        print(f"Accountant: {reply}\n")


if __name__ == "__main__":
    main()
