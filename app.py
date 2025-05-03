import streamlit as st
from resume_parser import extract_text_from_pdf
from scorer import get_resume_score
from utils import format_score

st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 AI Resume Screener")

st.subheader("📝 Enter Job Description")
job_description = st.text_area("Paste the job description here", height=200)

st.subheader("📂 Upload Resumes (PDF Only)")
uploaded_files = st.file_uploader("Upload one or more PDF resumes", type=["pdf"], accept_multiple_files=True)

if st.button("📊 Score Resumes"):
    if not job_description:
        st.warning("Please enter a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        results = []

        for file in uploaded_files:
            resume_text = extract_text_from_pdf(file)
            score = get_resume_score(resume_text, job_description)
            results.append({
                "Filename": file.name,
                "Score": score,
                "Rating": format_score(score)
            })

        results.sort(key=lambda x: x["Score"], reverse=True)

        st.success("✅ Resume screening completed!")
        st.dataframe(results)
