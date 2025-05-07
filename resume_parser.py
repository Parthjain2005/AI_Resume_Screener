from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import re
import spacy
from sentence_transformers import SentenceTransformer
from collections import Counter


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove special chars/numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text
def extract_sections(text, section_headers):
    sections = {}
    for header in section_headers:
        # Use regex to extract content under each section
        pass
    return sections


def keyword_match_score(resume_text, jd_text, top_n=10):
    tfidf = TfidfVectorizer(max_features=top_n)
    tfidf.fit([jd_text])  # Fit only on JD to extract top keywords
    keywords = tfidf.get_feature_names_out()
    resume_words = resume_text.split()
    keyword_counts = Counter(resume_words)
    score = sum([keyword_counts.get(keyword, 0) for keyword in keywords]) / len(keywords)
    return score

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_bert_score(resume_text, jd_text):
    resume_embedding = model.encode(resume_text)
    jd_embedding = model.encode(jd_text)
    score = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
    return score
def hybrid_score(resume_text, jd_text, alpha=0.5):
    tfidf_score = get_resume_score(resume_text, jd_text) / 100  # Normalize to [0,1]
    bert_score = get_bert_score(resume_text, jd_text)
    return alpha * tfidf_score + (1 - alpha) * bert_score

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]  # If trained NER model exists
    return skills