from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5-20251001")

system = SystemMessage(content="You are a helpful assistant.")
history = [system]


def chat(user_input: str) -> str:
    history.append(HumanMessage(content=user_input))
    response = model.invoke(history)
    history.append(AIMessage(content=response.content))
    return response.content


def main():
    print("Chatbot ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        reply = chat(user_input)
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()
