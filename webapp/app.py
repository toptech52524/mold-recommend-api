# webapp/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import csv, os, sys, base64, requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.recommend import load_data, train_vectorizer, recommend_designs

app = Flask(__name__)
CORS(app)

# 절대경로 1곳으로 고정 (webapp 폴더 기준 상위에 CSV)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.abspath(os.path.join(BASE_DIR, "../2025년 금형제작리스트_통합.csv"))

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO  = os.environ.get("GITHUB_REPO")
GITHUB_FILE  = os.environ.get("GITHUB_FILE")  # "2025년 금형제작리스트_통합.csv"

print("📂 CSV 데이터 로드 중...", CSV_PATH)
df = load_data(CSV_PATH)                         # ✅ 경로 넘김
vectorizer, tfidf_matrix = train_vectorizer(df)
print(f"✅ 데이터 로드 완료 ({len(df)}건)")

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

@app.route("/api/upload", methods=["POST"])
def upload_new_data():
    try:
        data = request.get_json()

        # 1) 로컬 CSV에 append (항상 CSV_PATH 한 곳!)
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                data.get("제번"), data.get("고객사"), data.get("설계"), data.get("제작처"),
                data.get("품명"), data.get("제품"), data.get("제품종류"), data.get("금형사이즈"),
                data.get("재질"), data.get("자재사이즈"), data.get("제품사이즈"),
                data.get("구조"), data.get("set_yn"), data.get("공정")
            ])

        # 2) GitHub 동기화 (있으면)
        if all([GITHUB_TOKEN, GITHUB_REPO, GITHUB_FILE]):
            with open(CSV_PATH, "rb") as f:
                content = base64.b64encode(f.read()).decode("utf-8")

            headers = {"Authorization": f"token {GITHUB_TOKEN}"}
            get_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE}"
            res = requests.get(get_url, headers=headers)
            sha = res.json().get("sha")

            payload = {
                "message": f"update CSV by API ({data.get('제번','unknown')})",
                "content": content,
                **({"sha": sha} if sha else {})   # 파일이 있으면 sha 포함
            }
            put_url = get_url
            res_put = requests.put(put_url, headers=headers, json=payload)

            # 👉 업로드 진단 로그 (Render 로그에서 확인)
            print("📡 GitHub PUT status:", res_put.status_code)
            try:
                print("📡 GitHub PUT body:", res_put.json())
            except Exception:
                print("📡 GitHub PUT body: <non-JSON>")

        # 3) 메모리 벡터 갱신 (항상 같은 CSV_PATH 로드)
        global df, vectorizer, tfidf_matrix
        df = load_data(CSV_PATH)                    # ✅ 경로 넘김
        vectorizer, tfidf_matrix = train_vectorizer(df)

        return jsonify({"status": "success", "message": "CSV 업데이트 및(선택) GitHub 저장 완료"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "records": len(df)})

@app.route("/api/debug", methods=["GET"])
def debug_info():
    return jsonify({
        "rows_in_memory": len(df),
        "last_row": df.tail(1).to_dict(orient="records")
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)