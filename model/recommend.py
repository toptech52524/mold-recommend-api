# model/recommend.py
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_parentheses(text: str) -> str:
    if pd.isna(text):
        return ""
    cleaned = re.sub(r"\(.*?\)", "", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def load_data(csv_path: str) -> pd.DataFrame:   # ✅ 경로를 인자로 받기
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["제품종류"] = df["제품종류"].fillna("").astype(str)
    df["품명"] = df["품명"].fillna("").astype(str)
    df["품명_정제"] = df["품명"].apply(clean_parentheses)
    df["search_key"] = (df["제품종류"] + " " + df["품명_정제"]).str.strip()
    return df

def train_vectorizer(df: pd.DataFrame):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(df["search_key"])
    return vectorizer, tfidf

def recommend_designs(query: str, vectorizer, tfidf_matrix, df: pd.DataFrame, top_n: int = 10):
    query = str(query or "").strip()
    if not query:
        return pd.DataFrame(columns=["제번", "고객사", "제품종류", "품명", "유사도"])
    query = re.sub(r"\(.*?\)", "", query).strip()
    qvec = vectorizer.transform([query])
    sim = cosine_similarity(qvec, tfidf_matrix).flatten()
    idx = sim.argsort()[-top_n:][::-1]
    result = df.loc[idx, ["제번", "고객사", "제품종류", "품명", "제품"]].copy()
    result.insert(4, "유사도", sim[idx].round(4))
    return result.reset_index(drop=True)