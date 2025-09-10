# backend/llm_backend.py
# Run with: uvicorn backend.llm_backend:app --reload --port 8000
# RUN FROM PROJECT ROOT

from fastapi import FastAPI
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Literal
from transformers import pipeline
import torch
import re
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# Pydantic models
# -----------------------------
class AssignmentItem(BaseModel):
    assignment: str
    due_date: str  # expected format: YYYY-MM-DD
    time: Optional[str] = None  # e.g. "14:00"

class Assignments(BaseModel):
    assignments: List[AssignmentItem]

class ExtractRequest(BaseModel):
    text: str  # raw input text from React

# -----------------------------
# Load model once
# -----------------------------
device = 0 if torch.backends.mps.is_available() else -1

generator = pipeline(
    "text-generation",
    model="google/gemma-3-270m-it",
    device=device,
    max_new_tokens=400
)

# -----------------------------
# Helper to build prompt
# -----------------------------
def create_prompt(assignments_input: str) -> str:
    print("inp",assignments_input)
    return f"""
You are an assistant that extracts assignments and their due dates from raw text.

Your task:
- Extract all assignments and quizzes into a JSON array.  
- Each item must have these keys:  
  - "assignment" (string)  
  - "due_date" (string, format "YYYY-MM-DD" where 2025 is the current year until the month is on or after Janurary which would then be the next year) 
  - "time" (optional string, format "HH:MM" or null, use military time and only include if time is given)
  
Rules:
- List the assignment names exactly as given, but remove words like "due", "by", "assigned".

Input text:
{assignments_input}
"""

# -----------------------------
# Pre-parser to extract JSON
# -----------------------------
def extract_json(raw_text: str) -> str:
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if match:
        return match.group(0)
    return "[]"

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Assignment Extractor API")

# Allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",                 # React dev server
        "http://localhost:3000/deadline_tracker" # mounted frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Endpoint
# -----------------------------
@app.post("/extract-assignments", response_model=Assignments)
def extract_assignments(req: ExtractRequest):
    prompt = create_prompt(req.text)
    raw_output = generator(prompt)[0]["generated_text"]
    print("Raw model output:\n", raw_output)

    json_text = extract_json(raw_output)
    print("Extracted JSON:\n", json_text)

    try:
        assignments_obj = Assignments.model_validate_json(
            f'{{"assignments": {json_text}}}'
        )
        return assignments_obj
    except ValidationError as e:
        return {"assignments": [], "error": str(e)}
