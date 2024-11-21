import streamlit as st
from helper import *

st.title("LeadScape")
if "chatbot" not in st.session_state:
    st.session_state.chatbot = Chatbot()

st.markdown("Paste your conversation here...")
user_input = st.text_area(label="text area", height=400, label_visibility="hidden")

if user_input:
    response = st.session_state.chatbot.run_chain(user_input)
    st.markdown("Response")
    st.write("\n" + response)
