import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

st.title("Academic Note Summarizer")
st.write("Paste your academic note below and click 'Summarize' to generate a concise summary.")

user_input = st.text_area("Your Academic Note", height=200)

if st.button("Summarize"):
    if user_input and user_input.strip():
        try:
            summary = summarizer(user_input, max_length=130, min_length=30, do_sample=False)
            st.success("Summary:")
            st.write(summary[0]['summary_text'])
        except Exception as e:
            st.error("Summarization failed. Try with a shorter input or check your text.")
    else:
        st.warning("Please enter some text to summarize.")
