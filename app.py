import streamlit as st

# App configuration
st.set_page_config(
    page_title="LLM Evaluation App",
    layout="wide"
)

# Main landing page content
st.title("Welcome to the LLM Evaluation App")
st.write("This app allows you to evaluate and interact with LLMs using two features:")
st.write("1. **Question/Answer Page** - Test and compare LLM-generated answers.")
st.write("2. **Chat Page** - Interact with the selected LLM in a conversational interface.")
st.sidebar.success("Select a page from the sidebar to begin!")
