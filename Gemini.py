import streamlit as st
from dotenv import load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains import LLMChain

load_dotenv()

# ---------- Custom CSS ----------
custom_css = """
<style>

body {
    background-color:#f7f7f7;
}

/* center the app */
.block-container {
    padding-top: 2rem;
}

/* Title styling */
h1 {
    text-align:center;
    background: linear-gradient(90deg, #005af0, #00e5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight:900;
}

/* chat bubble user */
.user-msg {
    background: #DCF8C6;
    color:#000;
    padding:12px 16px;
    border-radius:12px;
    margin-bottom:10px;
    width:fit-content;
    max-width:75%;
}

/* chat bubble bot */
.bot-msg {
    background: #ffffff;
    color:#000;
    padding:12px 16px;
    border-radius:12px;
    margin-bottom:10px;
    width:fit-content;
    max-width:75%;
    border:1px solid #eee;
}

/* input box styling */
.stTextInput > div > input {
    border-radius:10px;
    border:1px solid #444;
    padding:10px;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


# Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful chatbot."),
    HumanMessagePromptTemplate.from_template("{user_input}")
])

chain = LLMChain(llm=llm, prompt=prompt)

# ---------- Streamlit UI ----------
st.title("Gemini LLM Support Bot by Chinmay 🤖")

# Save conversation in session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show chat bubbles
for message, sender in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f'<div class="user-msg">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{message}</div>', unsafe_allow_html=True)

# User input box
input_text = st.chat_input("Type your message here...")

# When user sends
if input_text:
    st.session_state.chat_history.append((input_text, "user"))
    response = chain.run({"user_input": input_text})
    st.session_state.chat_history.append((response, "bot"))
    st.rerun()
