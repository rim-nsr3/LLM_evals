import streamlit as st
from pages.testing_LLM import fetch_llm_response, fetch_huggingface_response

# Streamlit App Setup
st.set_page_config(page_title="Chat with LLMs", layout="wide")
st.title("Chat Page")

# Step 1: Model Selection
st.subheader("1. Select an LLM Model")
available_models = ["Llama-3.1 LLM", "Mistral LLM"]
selected_model = st.selectbox("Choose a model for chatting:", available_models)

# Step 2: Chat Interface
st.subheader("2. Chat Interface")
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # Initialize chat history

# Display chat history
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").markdown(message["content"])

# Input box for user to type a message
user_input = st.chat_input("Type your message here...")
if user_input:
    # Append user message to chat history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # Generate response based on selected model
    with st.spinner("Generating response..."):
        if selected_model == "GROQ LLM":
            response = fetch_llm_response(user_input)  # Replace with your GROQ API function
        elif selected_model == "Hugging Face LLM":
            response = fetch_huggingface_response(user_input)  # Replace with Hugging Face function

    # Append assistant's response to chat history
    st.session_state["messages"].append({"role": "assistant", "content": response})

    # Display assistant's response
    st.chat_message("assistant").markdown(response)
