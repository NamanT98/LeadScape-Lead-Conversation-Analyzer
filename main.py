import streamlit as st
from helper import *

st.title("LeadScape")
if "chatbot" not in st.session_state:
    st.session_state.chatbot = PipelineBot()
if "submit" not in st.session_state:
    st.session_state.submit = False
if "score" not in st.session_state:
    st.session_state.score=None

st.markdown("Paste your conversation here...")
user_input = st.text_area(label="text area", height=400, label_visibility="hidden")

if st.button("Submit"):
    st.session_state.submit = True

if user_input and st.session_state.submit:
    st.session_state.chatbot.get_summary(user_input)
    st.session_state.score=st.session_state.chatbot.get_score(user_input)
    if st.session_state.score<=0.6:
        st.markdown(f"Score Received: {round(st.session_state.score,2)}, Modifying Response...")
        response = st.session_state.chatbot.run_chain(user_input)
        st.write("\n" + response)
    elif st.session_state.score>0.6:
        st.markdown(f"Score Received:{st.session_state.score}")
