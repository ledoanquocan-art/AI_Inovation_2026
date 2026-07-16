# app/database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./dti_database.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DTIEvaluationModel(Base):
    __tablename__ = "dti_evaluations"

    id = Column(Integer, primary_key=True, index=True)
    unit_name = Column(String, index=True)       # Tên đơn vị nộp
    pillar = Column(String)                      # Trụ cột DTI (Hạ tầng, Thể chế, ...)
    self_score = Column(Float)                   # Điểm tự đánh giá
    ai_score = Column(Float)                     # Điểm AI đề xuất sau khi đọc minh chứng
    evidence_file = Column(String)               # Đường dẫn file minh chứng đã lưu
    ai_comment = Column(Text)                    # Nhận xét từ AI
    status = Column(String, default="Pending")   # Trạng thái duyệt: Approved/Pending/Rejected

# Khởi tạo bảng dữ liệu
def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()