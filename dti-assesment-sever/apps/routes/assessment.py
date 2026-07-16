# app/routes/assessment.py
import os
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.database import get_db, DTIEvaluationModel
from app.ai_processor import analyze_evidence_with_ai

router = APIRouter()

# Thư mục lưu trữ file minh chứng tải lên
UPLOAD_DIR = "./uploaded_evidences"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/submit", response_model=None)
async def submit_and_evaluate(
    unit_name: str = Form(...),
    pillar: str = Form(...),
    self_score: float = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    API tiếp nhận hồ sơ đánh giá DTI và kích hoạt AI thẩm định minh chứng tự động.
    """
    # 1. Lưu file minh chứng tạm thời vào server
    file_path = os.path.join(UPLOAD_DIR, f"{unit_name}_{pillar}_{file.filename}".replace(" ", "_"))
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # 2. Gọi module AI để phân tích và chấm điểm minh chứng
    ai_result = analyze_evidence_with_ai(pillar, self_score, file_path)
    
    # 3. Tạo record mới lưu vào cơ sở dữ liệu
    db_record = DTIEvaluationModel(
        unit_name=unit_name,
        pillar=pillar,
        self_score=self_score,
        ai_score=ai_result.get("ai_score", 0.0),
        evidence_file=file_path,
        ai_comment=ai_result.get("comment", "Không có nhận xét."),
        status="Approved" if ai_result.get("ai_score", 0.0) >= self_score else "Pending"
    )
    
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
    return {
        "message": "Đã tiếp nhận hồ sơ và hoàn thành thẩm định bằng AI!",
        "record_id": db_record.id,
        "results": {
            "unit": db_record.unit_name,
            "pillar": db_record.pillar,
            "self_score": db_record.self_score,
            "ai_suggested_score": db_record.ai_score,
            "ai_verdict": db_record.ai_comment,
            "status": db_record.status
        }
    }

@router.get("/records")
def get_all_records(db: Session = Depends(get_db)):
    """Lấy danh sách tất cả các đơn vị đã đánh giá để hiển thị lên Dashboard"""
    records = db.query(DTIEvaluationModel).all()
    return records