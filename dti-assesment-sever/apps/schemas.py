# app/schemas.py
from pydantic import BaseModel
from typing import Optional

class DTISubmission(BaseModel):
    unit_name: str
    pillar: str
    self_score: float
    description: str

class DTIResponse(BaseModel):
    id: int
    unit_name: str
    pillar: str
    self_score: float
    ai_score: float
    evidence_file: Optional[str]
    ai_comment: str
    status: str

    class Config:
        from_attributes = True