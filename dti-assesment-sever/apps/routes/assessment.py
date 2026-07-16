# app/routes/assessment.py
import os
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.database import get_db, DTIEvaluationModel
from app.ai_processor import classify_evidence_with_pytorch

router = APIRouter()
UPLOAD_DIR = "./uploaded_evidences"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/submit")
async def submit_and_evaluate(
    unit_name: str = Form(...),
    pillar: str = Form(...),          # Trụ cột người dùng tự chọn khai báo
    self_score: float = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. Lưu file minh chứng
    file_path = os.path.join(UPLOAD_DIR, f"{unit_name}_{file.filename}".replace(" ", "_"))
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # 2. Chạy phân loại thông minh bằng PyTorch
    pytorch_result = classify_evidence_with_pytorch(file_path)
    
    detected_pillar = pytorch_result["predicted_pillar"]
    confidence = pytorch_result["confidence"]
    ai_comment = pytorch_result["comment"]
    
    # 3. Logic đối chiếu nghiệp vụ:
    # Nếu hệ thống PyTorch phát hiện nội dung tài liệu lệch hoàn toàn với Trụ cột khai báo
    if detected_pillar != "Chưa xác định" and detected_pillar.lower() != pillar.lower():
        status = "Rejected"
        ai_score = round(self_score * 0.4, 1) # Phạt điểm vì nộp sai minh chứng
        verdict = f"CẢNH BÁO: Bạn khai báo cột '{pillar}' nhưng PyTorch phát hiện minh chứng thuộc về nhóm '{detected_pillar}' (Độ tin cậy {confidence}%). {ai_comment}"
    else:
        status = "Approved"
        ai_score = self_score
        verdict = f"Hợp lệ! PyTorch xác nhận minh chứng khớp với nhóm '{pillar}' (Độ tin cậy {confidence}%). {ai_comment}"
    
    # 4. Lưu vào Database SQLite
    db_record = DTIEvaluationModel(
        unit_name=unit_name,
        pillar=pillar,
        self_score=self_score,
        ai_score=ai_score,
        evidence_file=file_path,
        ai_comment=verdict,
        status=status
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
    return {
        "message": "Đã chạy thẩm định tài liệu bằng mô hình PyTorch cục bộ thành công!",
        "record_id": db_record.id,
        "ai_analysis": {
            "predicted_pillar_by_pytorch": detected_pillar,
            "confidence_level": f"{confidence}%",
            "decision": status,
            "final_score": ai_score,
            "reason": verdict
        }
    }

@router.get("/records")
def get_all_records(db: Session = Depends(get_db)):
    return db.query(DTIEvaluationModel).all()