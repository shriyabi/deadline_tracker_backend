from pydantic import BaseModel, ValidationError
from typing import List, Optional

class Assignments(BaseModel):
    assignment: str
    due_date: str
    due_time: Optional[str] = None