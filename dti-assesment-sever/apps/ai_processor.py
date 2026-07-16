# app/ai_processor.py
import os
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Cấu hình API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("⚠️ WARNING: GEMINI_API_KEY chưa được cấu hình trong file .env!")

def extract_text_from_pdf(file_path: str) -> str:
    """Đọc văn bản từ file PDF minh chứng"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Lỗi đọc file: {str(e)}"

def analyze_evidence_with_ai(pillar: str, self_score: float, file_path: str) -> dict:
    """Dùng Gemini AI đánh giá tính xác thực của tài liệu minh chứng"""
    
    # 1. Trích xuất text từ file minh chứng
    evidence_text = extract_text_from_pdf(file_path)
    if not evidence_text.strip():
        return {
            "ai_score": 0.0,
            "comment": "Không thể trích xuất thông tin từ file minh chứng được tải lên. Vui lòng kiểm tra lại định dạng file."
        }

    # 2. Tạo prompt ngữ cảnh chuyên sâu về DTI của Bộ KH&CN
    prompt = f"""
    Bạn là một chuyên gia đánh giá Chuyển đổi số độc lập làm việc cho Bộ Khoa học và Công nghệ Việt Nam.
    Hãy thẩm định báo cáo minh chứng của đơn vị gửi lên theo thông tin sau:
    
    - Trụ cột đánh giá: {pillar}
    - Điểm đơn vị tự chấm (thang điểm 10): {self_score}
    
    Nội dung báo cáo minh chứng thực tế trích xuất từ file đính kèm:
    ---
    {evidence_text[:3000]}  # Giới hạn 3000 ký tự đầu tiên để tránh tràn ngữ cảnh
    ---
    
    NHIỆM VỤ CỦA BẠN:
    1. Đánh giá xem nội dung minh chứng trên có khớp và đủ cơ sở pháp lý/kỹ thuật chứng minh cho điểm số tự chấm ({self_score}/10) ở trụ cột {pillar} không.
    2. Đưa ra điểm số đề xuất (thang điểm 10) dựa trên mức độ hoàn thành minh chứng thực tế.
    3. Nhận xét chi tiết (ngắn gọn dưới 100 từ), chỉ rõ điểm hợp lệ và điểm còn thiếu trong minh chứng.
    
    YÊU CẦU TRẢ VỀ kết quả theo định dạng JSON chuẩn như sau (Không viết thêm bất kỳ từ nào ngoài JSON):
    {{
        "ai_score": <điền_số_điểm_đề_xuất_từ_0_đến_10>,
        "comment": "<nhận_xét_chuyên_môn_của_bạn>"
    }}
    """

    try:
        # Gọi model Gemini 1.5 Flash (Xử lý cực nhanh và tối ưu chi phí/tốc độ cho Hackathon)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        # Parse kết quả JSON trả về
        import json
        result = json.loads(response.text)
        return result
    except Exception as e:
        return {
            "ai_score": round(self_score * 0.8, 1), # Fallback nếu lỗi API
            "comment": f"[Hệ thống tự động chấm điểm] Đã ghi nhận minh chứng. Lỗi phân tích AI: {str(e)}"
        }