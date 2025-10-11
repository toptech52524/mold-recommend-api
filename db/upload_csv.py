# db/upload_csv.py
import pandas as pd
import numpy as np
from db.connect import get_connection

TABLE = "designs"

# CSV의 컬럼명 -> DB 컬럼명 매핑 (SET → set_yn 주의)
CSV_TO_DB_COLS = [
    ("제번", "제번"),
    ("고객사", "고객사"),
    ("설계", "설계"),
    ("제작처", "제작처"),
    ("품명", "품명"),
    ("제품", "제품"),
    ("제품종류", "제품종류"),
    ("금형사이즈", "금형사이즈"),
    ("재질", "재질"),
    ("자재사이즈", "자재사이즈"),
    ("제품사이즈", "제품사이즈"),
    ("구조", "구조"),
    ("SET", "set_yn"),
    ("공정", "공정"),
    # CSV에 이미지 파일명이 비어있다면 None으로 넣음
    # ("image_path","image_path")  # 필요시 CSV에 컬럼 추가 후 사용
]

def _to_none(x):
    """NaN/빈문자 → None 변환 (MySQL INSERT 호환)"""
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s != "" else None

def upload_csv_to_mysql(csv_path: str, encoding: str = "utf-8-sig", chunksize: int = 0):
    # 1) CSV 로드
    df = pd.read_csv(csv_path, encoding=encoding)
    # 필요한 컬럼만 슬라이스(없으면 자동 생성해서 None)
    for csv_col, _ in CSV_TO_DB_COLS:
        if csv_col not in df.columns:
            df[csv_col] = np.nan

    # 2) 값 정리
    for csv_col, _ in CSV_TO_DB_COLS:
        df[csv_col] = df[csv_col].map(_to_none)

    # image_path 입력이 없다면 None으로
    if "image_path" not in df.columns:
        df["image_path"] = None

    # 3) INSERT 준비
    db_cols = [db for _, db in CSV_TO_DB_COLS] + ["image_path"]
    placeholders = ", ".join(["%s"] * len(db_cols))
    colnames_sql = ", ".join(db_cols)
    insert_sql = f"INSERT INTO {TABLE} ({colnames_sql}) VALUES ({placeholders})"

    values = df[[c for c, _ in CSV_TO_DB_COLS] + ["image_path"]].values.tolist()

    # 4) DB 연결 및 업로드
    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()
    try:
        if chunksize and chunksize > 0:
            # 대용량일 경우 배치로
            for i in range(0, len(values), chunksize):
                cur.executemany(insert_sql, values[i:i+chunksize])
        else:
            cur.executemany(insert_sql, values)
        conn.commit()
        print(f"✅ 업로드 완료: {len(values)}건")
    except Exception as e:
        conn.rollback()
        print("❌ 업로드 실패:", e)
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    # 프로젝트 루트에 CSV를 복사해 두었으면 아래 경로 사용
    upload_csv_to_mysql("2025년 금형제작리스트_통합.csv")
