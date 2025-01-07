import altair as alt  # Altair for visualizations
import os
from openai import OpenAI
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity  # For cosine similarity
from sklearn.feature_extraction.text import CountVectorizer  # To vectorize text
import pandas as pd  # To handle dataframes
import numpy as np  # For numerical operations
import requests
import textdistance
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")  # Fetch API key from environment variables
)

# Function to fetch response from the LLM (e.g., GROQ API)
def fetch_llm_response(question):
    """Fetches the answer from an LLM (e.g., GROQ API)."""
    try:
        # System prompt for context
        system_prompt = """You are an advanced LLM. Answer the question clearly and concisely."""

        # Augmented query for LLM
        augmented_query = (
            "<CONTEXT>\n" +
            "\n-------\n" + question + "\n-------\n</CONTEXT>\n\nMY QUESTION:\n" +
            question
        )

        # API request
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Ensure this model is supported by the GROQ API
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ]
        )

        # Extract the content of the response
        # Corrected to use `response.choices` instead of subscript notation
        return response.choices[0].message.content

    except Exception as e:
        # Return a meaningful error message if something goes wrong
        return f"An error occurred: {str(e)}"

def fetch_huggingface_response(question, model="bigscience/bloom"):
    """Fetches the answer from Hugging Face Inference API."""
    try:
        url = f"https://api-inference.huggingface.co/models/mistralai/Mistral-Nemo-Instruct-2407"
        headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
        payload = {"inputs": question}
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            return f"Error: {response.status_code}, {response.json()}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


# Streamlit App Setup
st.set_page_config(page_title="LLM Question Answering", layout="wide")
st.title("Question Answer Page")

# Input Section
st.subheader("1. Ask a Question")
with st.form("question_form"):
    question = st.text_area("Enter your question:", height=100)
    expected_answer = st.text_area("Enter the expected answer:", height=100)
    submitted = st.form_submit_button("Get Answer")

# LLM Answer Section
if submitted:
    st.subheader("2. LLM Answers")
    if question.strip():
        st.write(f"**Your Question:** {question}")
        st.write(f"**Expected Answer:** {expected_answer if expected_answer else 'None provided'}")

        with st.spinner("Fetching answer from Llama 3.1.8b-instant LLM..."):
            llm_answer_groq = fetch_llm_response(question)

        with st.spinner("Fetching answer from Mistral-Nemo-Instruct-2407 LLM..."):
            llm_answer_hf = fetch_huggingface_response(question, model="bigscience/bloom")  # Or "google/flan-t5-large"

        st.write("**GROQ LLM's Answer:**")
        st.success(llm_answer_groq)
        st.write("**Hugging Face LLM's Answer:**")
        st.success(llm_answer_hf)

        # Store both answers for metrics comparison
        answers = {"GROQ LLM": llm_answer_groq, "Hugging Face LLM": llm_answer_hf}
    else:
        st.error("Please enter a valid question.")

# Metrics Section
if submitted and expected_answer.strip():
    st.subheader("3. Similarity Metrics")

    # Similarity for each model
    similarity_data = []
    for model_name, model_answer in answers.items():
        # Cosine Similarity
        texts = [expected_answer, model_answer]
        vectorizer = CountVectorizer().fit_transform(texts)
        vectors = vectorizer.toarray()
        cosine_sim = cosine_similarity(vectors)[0, 1]

        # Jaccard Similarity
        jaccard_sim = textdistance.jaccard(expected_answer, model_answer)

        # Append data for visualization
        similarity_data.append({"Model": model_name, "Metric": "Cosine Similarity", "Value": cosine_sim})
        similarity_data.append({"Model": model_name, "Metric": "Jaccard Similarity", "Value": jaccard_sim})

    # Convert to DataFrame
    similarity_df = pd.DataFrame(similarity_data)

    # Display Metrics
    st.write("### Similarity Metrics Table")
    st.table(similarity_df)

    # Visualization
    st.subheader("4. Visualizations")

    # Create a bar chart
    similarity_chart = alt.Chart(similarity_df).mark_bar().encode(
        x="Model",
        y="Value",
        color="Metric",
        column="Metric"
    ).properties(title="Similarity Metrics by Model")
    st.altair_chart(similarity_chart, use_container_width=True)
