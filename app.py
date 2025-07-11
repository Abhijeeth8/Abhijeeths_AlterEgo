import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from main import get_retriever, ret_aug_gen, exec_tools, llm_with_tools

import json

creds_dict = {
    "installed": {
        "client_id": st.secrets["CLIENT_ID"],
        "project_id": "alterego-463605",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": st.secrets["CLIENT_SECRET"],
        "redirect_uris": ["http://localhost"]
    }
}

with open("credentials.json", "w") as f:
    json.dump(creds_dict, f)

creds_dict = {
    "token": st.secrets["TOKEN"],
    "refresh_token": st.secrets["REFRESH_TOKEN"],
    "token_uri": "https://oauth2.googleapis.com/token",
    "client_id": st.secrets["CLIENT_ID"],
    "client_secret": st.secrets["CLIENT_SECRET"],
    "scopes": ["https://www.googleapis.com/auth/gmail.compose"],
    "universe_domain": "googleapis.com",
    "account": "",
    "expiry": "2025-07-10T05:08:34Z"
}

with open("token.json", "w") as f:
    json.dump(creds_dict, f)

st.set_page_config(page_title="Alterego", layout="wide")
st.title("🤖 Abhijeeth's Alterego")

# CSS Styling
st.markdown("""
    <style>
    .user-msg {
        background-color: #F8E7DA;
        padding: 10px 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: black;
    }
    .bot-msg {
        background-color: #E1A95F;
        padding: 10px 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        color:black;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = get_retriever("resume_vectorstore")

prompt_map = {
    "Education": "What is your educational background?",
    "Experience": "Tell me about your work experience.",
    "Projects": "What projects have you worked on?",
    "Skills": "What are your key technical skills?"
}

prompt = st.chat_input("Ask something about me...")

# Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("You"):
            st.markdown(f"<div class='user-msg'>{msg.content}</div>", unsafe_allow_html=True)
    else:
        with st.chat_message("Abhijeeth"):
            st.markdown(f"<div class='bot-msg'>{msg.content}</div>", unsafe_allow_html=True)

# User input
if prompt:
    with st.chat_message("You"):
        st.markdown(f"<div class='user-msg'>{prompt}</div>", unsafe_allow_html=True)

    st.session_state.chat_history.append(HumanMessage(prompt))

    response = ret_aug_gen(st.session_state.retriever, prompt, st.session_state.chat_history)
    st.session_state.chat_history.append(response)

    # Handle tool execution
    if response.tool_calls:
        tool_response = exec_tools(llm_with_tools, response)
        st.session_state.chat_history.append(tool_response)

        final_response = llm_with_tools.invoke(st.session_state.chat_history)
        st.session_state.chat_history.append(final_response)

        with st.chat_message("Abhijeeth"):
            st.markdown(f"<div class='bot-msg'>{final_response.content}</div>", unsafe_allow_html=True)
    else:
        with st.chat_message("Abhijeeth"):
            st.markdown(f"<div class='bot-msg'>{response.content}</div>", unsafe_allow_html=True)