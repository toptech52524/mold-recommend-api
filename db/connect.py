# db/connect.py
import mysql.connector

def get_connection():
    """
    MySQL Local instance(MySQL80) 접속.
    root 계정 비밀번호가 없다면 password="" 그대로 두세요.
    현재 비밀번호는 작성자의 비밀번호로 임의로 작성되어 있으니 사용 시, 작업자의 환경애 맞게 수정하여 쓰면 됩니다
    """
    return mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="1234",          # 현재 로컬 비밀번호 설정 1234 인데 환경에 맞게 수정
        database="mold_db",
        charset="utf8mb4"
    )