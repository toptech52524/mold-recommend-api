from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import csv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.recommend import load_data, train_vectorizer, recommend_designs

app = Flask(__name__)
CORS(app)

# CSV 경로 설정 (Render 환경 호환)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "../2025년 금형제작리스트_통합.csv")

# ----------------------------------------
# 초기 데이터 로드
# ----------------------------------------
print("📂 CSV 데이터 로드 중...")
df = load_data() 
vectorizer, tfidf_matrix = train_vectorizer(df)
print(f"✅ 데이터 로드 완료 ({len(df)}건)")

# ----------------------------------------
# 추천 API
# ----------------------------------------
@app.route("/api/recommend", methods=["POST"])
def recommend_api():
    try:
        data = request.get_json()
        product_type = data.get("product_type", "")
        product_name = data.get("product_name", "")
        query = f"{product_type} {product_name}".strip()
        result = recommend_designs(query, vectorizer, tfidf_matrix, df)
        return jsonify(result.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------------------
# CSV 데이터 추가 API
# ----------------------------------------
@app.route("/api/upload", methods=["POST"])
def upload_new_data():
    try:
        data = request.get_json()

        # CSV 파일에 새 데이터 추가
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                data.get("제번"), data.get("고객사"), data.get("설계"), data.get("제작처"),
                data.get("품명"), data.get("제품"), data.get("제품종류"), data.get("금형사이즈"),
                data.get("재질"), data.get("자재사이즈"), data.get("제품사이즈"),
                data.get("구조"), data.get("set_yn"), data.get("공정")
            ])

        # TF-IDF 벡터 갱신
        global df, vectorizer, tfidf_matrix
        df = pd.read_csv(CSV_PATH, encoding="utf-8")
        vectorizer, tfidf_matrix = train_vectorizer(df)

        return jsonify({"status": "success", "message": "CSV에 새 데이터 추가 및 벡터 갱신 완료"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------------------
# 서버 상태 확인
# ----------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "records": len(df)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)