import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot")

model = ChatAnthropic(model="claude-haiku-4-5-20251001")

if "history" not in st.session_state:
    st.session_state.history = [SystemMessage(content="You are a helpful assistant.")]

for msg in st.session_state.history[1:]:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

if prompt := st.chat_input("Type a message..."):
    st.session_state.history.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.invoke(st.session_state.history)
        st.write(response.content)

    st.session_state.history.append(AIMessage(content=response.content))
