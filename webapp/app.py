from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import csv
import os
import sys
import base64
import requests

# -----------------------------------------------------
# 경로 설정 및 모듈 import
# -----------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.recommend import load_data, train_vectorizer, recommend_designs

app = Flask(__name__)
CORS(app)

# -----------------------------------------------------
# 환경 변수 (Render → Environment Variables 에 설정 필요)
# -----------------------------------------------------
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO")  # 예: Hyeonseo-OH/mold-recommend-api
GITHUB_FILE = os.environ.get("GITHUB_FILE")  # 예: 2025년 금형제작리스트_통합.csv

# CSV 파일 경로 (Render 내부)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "../2025년 금형제작리스트_통합.csv")

# -----------------------------------------------------
# 초기 데이터 로드
# -----------------------------------------------------
print("📂 CSV 데이터 로드 중...")
df = load_data()
vectorizer, tfidf_matrix = train_vectorizer(df)
print(f"✅ 데이터 로드 완료 ({len(df)}건)")

# -----------------------------------------------------
# 추천 API
# -----------------------------------------------------
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

# -----------------------------------------------------
# CSV 데이터 업로드 + GitHub 반영
# -----------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def upload_new_data():
    try:
        data = request.get_json()

        # Render 내부 CSV에 새 행 추가
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                data.get("제번"), data.get("고객사"), data.get("설계"), data.get("제작처"),
                data.get("품명"), data.get("제품"), data.get("제품종류"), data.get("금형사이즈"),
                data.get("재질"), data.get("자재사이즈"), data.get("제품사이즈"),
                data.get("구조"), data.get("set_yn"), data.get("공정")
            ])

        # CSV를 GitHub로 업로드 (Base64 인코딩)
        if not all([GITHUB_TOKEN, GITHUB_REPO, GITHUB_FILE]):
            return jsonify({"error": "GitHub 환경변수가 설정되지 않았습니다."}), 500

        with open(CSV_PATH, "rb") as f:
            content = base64.b64encode(f.read()).decode("utf-8")

        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        get_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE}"
        res = requests.get(get_url, headers=headers).json()
        sha = res.get("sha")  # 기존 파일 SHA

        payload = {
            "message": f"update CSV by API ({data.get('제번', 'unknown')})",
            "content": content,
            "sha": sha
        }

        put_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE}"
        res_put = requests.put(put_url, headers=headers, json=payload)

        # TF-IDF 벡터 갱신
        global df, vectorizer, tfidf_matrix
        df = load_data()
        vectorizer, tfidf_matrix = train_vectorizer(df)

        return jsonify({
            "status": "success",
            "message": "CSV 업데이트 및 GitHub 저장 완료",
            "github_response": res_put.json()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------------------------------
# 서버 상태 확인 API
# -----------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "records": len(df)})

# -----------------------------------------------------
# (선택) 디버그용 데이터 확인 API
# -----------------------------------------------------
@app.route("/api/debug", methods=["GET"])
def debug_info():
    return jsonify({
        "rows_in_memory": len(df),
        "last_row": df.tail(1).to_dict(orient="records")
    })

# -----------------------------------------------------
# Flask 실행
# -----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
