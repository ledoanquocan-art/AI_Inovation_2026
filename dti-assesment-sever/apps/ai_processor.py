# app/ai_processor.py
import os
import torch
import torch.nn as nn
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Định nghĩa kiến trúc mạng nơ-ron phân loại bằng PyTorch
class DTITextClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(DTITextClassifier, self).__init__()
        # Mạng Feedforward đơn giản: Input -> Hidden (128) -> ReLU -> Dropout -> Output (4 classes)
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# 2. Giả lập bộ dữ liệu nhỏ để "huấn luyện" nhanh Vectorizer (Trong Hackathon cần chạy mượt ngay)
# Từ ngữ đặc trưng cho các trụ cột DTI của Bộ KH&CN
categories = ["Thể chế số", "Hạ tầng số", "Nhân lực số", "An toàn thông tin"]
training_corpus = [
    "quyết định ban hành kế hoạch chính sách đề án chuyển đổi số thể chế chỉ đạo",
    "máy chủ đám mây hạ tầng mạng cáp quang trung tâm dữ liệu thiết bị máy tính đường truyền",
    "đào tạo tập huấn nhân lực kỹ năng số kỹ sư công nghệ chuyên gia khóa học bồi dưỡng",
    "bảo mật an toàn thông tin mã hóa tường lửa diệt virus tấn công mạng lỗ hổng giám sát"
]

# Sử dụng TF-IDF để chuyển văn bản tiếng Việt thành vector số
vectorizer = TfidfVectorizer(max_features=500)
vectorizer.fit(training_corpus)

# Khởi tạo mô hình PyTorch với số lượng đặc trưng đầu vào là 500
input_dim = 500
model = DTITextClassifier(input_dim=input_dim, num_classes=4)
model.eval() # Chuyển sang chế độ dự báo (Inference)

def extract_text_from_pdf(file_path: str) -> str:
    """Đọc văn bản từ file PDF minh chứng"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return ""

def classify_evidence_with_pytorch(file_path: str) -> dict:
    """Sử dụng mô hình PyTorch để phân loại xem file minh chứng thuộc trụ cột nào"""
    text = extract_text_from_pdf(file_path)
    if not text.strip():
        return {
            "predicted_pillar": "Chưa xác định",
            "confidence": 0.0,
            "comment": "File trống hoặc không đọc được dữ liệu."
        }
    
    # Tiền xử lý: Chuyển văn bản thành Vector số TF-IDF
    text_vector = vectorizer.transform([text.lower()]).toarray()
    text_tensor = torch.tensor(text_vector, dtype=torch.float32)
    
    # Đưa qua mô hình PyTorch dự báo không tính toán đạo hàm (no_grad)
    with torch.no_grad():
        outputs = model(text_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class_idx = torch.max(probabilities, dim=1)
        
    predicted_idx = predicted_class_idx.item()
    predicted_pillar = categories[predicted_idx]
    confidence_score = round(confidence.item() * 100, 2)
    
    # Gợi ý dựa trên kết quả phân loại của PyTorch
    comments = {
        0: "Tài liệu tập trung nhiều vào cơ sở pháp lý, thể chế chỉ đạo ban hành văn bản của đơn vị.",
        1: "Tài liệu cung cấp minh chứng rõ ràng về đầu tư hạ tầng phần cứng, mạng và máy chủ.",
        2: "Tài liệu cho thấy đơn vị chú trọng tổ chức đào tạo, nâng cao trình độ nhân sự số.",
        3: "Tài liệu hợp lệ về mặt kiểm thử an toàn thông tin, an ninh mạng hệ thống."
    }
    
    return {
        "predicted_pillar": predicted_pillar,
        "confidence": confidence_score,
        "comment": comments.get(predicted_idx, "Đã phân tích xong tài liệu.")
    }