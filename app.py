


import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

st.set_page_config(page_title="Resume Job Match Scorer", page_icon="📄", layout="wide")

st.markdown("""
Upload your resume (PDF) and paste a job description to see how well they match!
This tool uses **TF-IDF + Cosine Similarity**.
""")

with st.sidebar:
    st.header("About")
    st.info("Resume–Job matching using NLP")

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    return " ".join(w for w in words if w not in stop_words)

def calculate_similarity(resume, job):
    resume = remove_stopwords(clean_text(resume))
    job = remove_stopwords(clean_text(job))
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume, job])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100
    return round(score, 2)

def main():
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_desc = st.text_area("Paste Job Description")

    if st.button("Analyze Match"):
        if resume_file and job_desc:
            resume_text = extract_text_from_pdf(resume_file)
            score = calculate_similarity(resume_text, job_desc)
            st.metric("Match Score", f"{score}%")
        else:
            st.warning("Upload resume and paste job description")

if __name__ == "__main__":
    main()
