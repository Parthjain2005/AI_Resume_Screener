from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from resume_parser import *
# def get_resume_score(resume_text, jd_text):
#     vectorizer = TfidfVectorizer()
#     vectors = vectorizer.fit_transform([resume_text, jd_text])
#     score = cosine_similarity(vectors[0:1], vectors[1:2])
#     return round(score[0][0] * 100, 2)

def get_resume_score(resume_text, jd_text):
    # Preprocess
    resume_clean = preprocess(resume_text)
    jd_clean = preprocess(jd_text)
    
    # Keyword score
    keyword_score = keyword_match_score(resume_clean, jd_clean)
    
    # BERT score
    bert_score = get_bert_score(resume_clean, jd_clean)
    
    # Hybrid score (adjust weights)
    final_score = 0.4 * keyword_score + 0.6 * bert_score
    return round(final_score * 100, 2)


