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
    data = request.get_json()# webapp/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from db.connect import get_connection
from model.recommend import load_data, train_vectorizer, recommend_designs
import threading
import time

app = Flask(__name__)
CORS(app)

# ====== 전역 캐시 ======
print("데이터 로드 중...")
df = load_data()
vectorizer, tfidf = train_vectorizer(df)
last_update = time.time()
print(f"데이터 로드 완료 ({len(df)}건)")

# ====== 추천 API ======
@app.route("/api/recommend", methods=["POST"])
def recommend_api():
    global df, vectorizer, tfidf
    try:
        data = request.get_json()
        product_type = data.get("product_type", "")
        product_name = data.get("product_name", "")
        query = f"{product_type} {product_name}".strip()

        results = recommend_designs(query, vectorizer, tfidf, df)
        return jsonify(results.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ====== 업로드 API ======
@app.route("/api/upload", methods=["POST"])
def upload_design():
    """
    외주에서 새로운 데이터를 업로드할 때 호출.
    DB에 추가 후 자동으로 추천 데이터셋 갱신.
    """
    global df, vectorizer, tfidf, last_update
    try:
        data = request.get_json()

        conn = get_connection()
        cursor = conn.cursor()

        sql = """
        INSERT INTO designs
        (제번, 고객사, 설계, 제작처, 품명, 제품, 제품종류,
         금형사이즈, 재질, 자재사이즈, 제품사이즈, 구조, set_yn, 공정)
        VALUES (%(제번)s, %(고객사)s, %(설계)s, %(제작처)s, %(품명)s, %(제품)s, %(제품종류)s,
                %(금형사이즈)s, %(재질)s, %(자재사이즈)s, %(제품사이즈)s, %(구조)s, %(set_yn)s, %(공정)s)
        """

        cursor.execute(sql, data)
        conn.commit()
        cursor.close()
        conn.close()

        # 데이터 갱신을 백그라운드에서 비동기로 실행
        def refresh_data():
            global df, vectorizer, tfidf, last_update
            print("🔄 DB 변경 감지 → 추천 모델 갱신 중...")
            new_df = load_data()
            new_vectorizer, new_tfidf = train_vectorizer(new_df)
            df, vectorizer, tfidf = new_df, new_vectorizer, new_tfidf
            last_update = time.time()
            print(f"✅ 모델 갱신 완료 ({len(df)}건)")

        threading.Thread(target=refresh_data).start()

        return jsonify({"message": "✅ 데이터 업로드 성공, 추천 목록에 반영 중"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ====== 헬스체크용 ======
@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status": "running",
        "records": len(df),
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_update))
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

    product_type = data.get("product_type", "")
    product_name = data.get("product_name", "")

    query = f"{product_type} {product_name}".strip()
    results = recommend_designs(query, vectorizer, tfidf, df, top_n=10)

    # DataFrame → dict 변환
    return jsonify(results.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)