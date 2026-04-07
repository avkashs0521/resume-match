from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np

model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def compute_similarity(resumes, jobs):
    r_text = [str(r["text"]).lower() for r in resumes]

    # include skills + boost them
    j_text = [
        (j["description"] + " " + " ".join(j.get("skills_required", []) * 3)).lower()
        for j in jobs
    ]

    # ---------- TF-IDF ----------
    corpus = r_text + j_text
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(corpus)

    r_tfidf = tfidf[:len(r_text)]
    j_tfidf = tfidf[len(r_text):]

    tfidf_sim = cosine_similarity(j_tfidf, r_tfidf)

    # ---------- Transformer ----------
    model = get_model()

    r_emb = model.encode(r_text)
    j_emb = model.encode(j_text)

    emb_sim = cosine_similarity(j_emb, r_emb)

    # ---------- FINAL SCORE ----------
    final_sim = 0.9 * tfidf_sim + 0.1 * emb_sim

    return final_sim