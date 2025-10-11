# main.py
from model.recommend import load_data, train_vectorizer, recommend_designs

def main():
    print("DB에서 데이터 로드…")
    df = load_data()
    print(f"   → {len(df)}건 로드")

    print("TF-IDF 학습…")
    vectorizer, tfidf = train_vectorizer(df)
    print("준비 완료")

    while True:
        print("\n------------------------------------")
        product_type = input("제품종류 입력 (예: 사출금형, 종료는 q): ").strip()
        if product_type.lower() in {"q", "quit", "exit"}:
            print("종료합니다.")
            break

        product_name = input("품명 입력 (예: 핸들커버): ").strip()
        if product_name.lower() in {"q", "quit", "exit"}:
            print("종료합니다.")
            break

        # 두 입력을 결합
        query = f"{product_type} {product_name}".strip()

        recs = recommend_designs(query, vectorizer, tfidf, df, top_n=10)
        if recs.empty:
            print("⚠️ 결과가 없습니다.")
        else:
            print("\n🔍 추천 결과 (상위 10)")
            print(recs.to_string(index=False))

if __name__ == "__main__":
    main()