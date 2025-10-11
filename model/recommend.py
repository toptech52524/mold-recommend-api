# model/recommend.py
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from db.connect import get_connection

def clean_parentheses(text: str) -> str:
    """괄호() 안의 내용 제거 + 공백 정리"""
    if pd.isna(text):
        return ""
    cleaned = re.sub(r"\(.*?\)", "", str(text))   # 괄호 및 내부 내용 제거
    cleaned = re.sub(r"\s+", " ", cleaned).strip() # 다중 공백 제거
    return cleaned

def load_data() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM designs", conn)
    conn.close()

    # 결측치 처리
    df["제품종류"] = df["제품종류"].fillna("").astype(str)
    df["품명"] = df["품명"].fillna("").astype(str)

    # 품명에서 괄호 내용 제거
    df["품명_정제"] = df["품명"].apply(clean_parentheses)

    # 검색 키 생성 (제품종류 + 정제된 품명)
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

    # 쿼리에서도 괄호 내용 제거 (입력 일관성 유지)
    query = re.sub(r"\(.*?\)", "", query).strip()

    qvec = vectorizer.transform([query])
    sim = cosine_similarity(qvec, tfidf_matrix).flatten()
    idx = sim.argsort()[-top_n:][::-1]

    # ✅ 출력 컬럼: 제번, 고객사, 품명, 제품, 제품종류
    result = df.loc[idx, ["제번", "고객사", "품명", "제품", "제품종류"]].copy()

    # 유사도 정보 추가 (맨 끝에)
    result["유사도"] = sim[idx].round(4)

    return result.reset_index(drop=True)