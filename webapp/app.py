# webapp/app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model.recommend import load_data, train_vectorizer, recommend_designs

app = Flask(__name__)
CORS(app)  # 프론트엔드에서 접근 허용

# 서버 시작 시 데이터/모델 로드
print("데이터 로드 중...")
df = load_data()
vectorizer, tfidf = train_vectorizer(df)
print(f"데이터 로드 완료 ({len(df)}건)")


# 기본 페이지 (테스트용)
@app.route("/")
def home():
    return render_template("index.html")


# API 엔드포인트
@app.route("/api/recommend", methods=["POST"])
def recommend_api():
    """
    요청 예시:
    {
        "product_type": "사출금형",
        "product_name": "핸들커버"
    }
    """
    data = request.get_json()
    product_type = data.get("product_type", "")
    product_name = data.get("product_name", "")

    query = f"{product_type} {product_name}".strip()
    results = recommend_designs(query, vectorizer, tfidf, df, top_n=10)

    # DataFrame → dict 변환
    return jsonify(results.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)