import streamlit as st
from helper import *

st.title("LeadScape")
if "chatbot" not in st.session_state:
    st.session_state.chatbot = Chatbot()
if "submit" not in st.session_state:
    st.session_state.submit = False

st.markdown("Paste your conversation here...")
user_input = st.text_area(label="text area", height=400, label_visibility="hidden")

if st.button("Submit"):
    st.session_state.submit = True

if user_input and st.session_state.submit:
    response = st.session_state.chatbot.run_chain(user_input)
    st.markdown("Response")
    st.write("\n" + response)
