import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class DateTimeModel(BaseModel):
    date:str=Field(description="Properly formatted date", pattern=r'^\d{2}-\d{2}-\d{4} \d{2}:\d{2}$')
    
    @field_validator("date")
    def check_format_date(cls, v):
        if not re.match(r'^\d{2}-\d{2}-\d{4} \d{2}:\d{2}$', v):  # Ensures 'DD-MM-YYYY HH:MM' format
            raise ValueError("The date should be in format 'DD-MM-YYYY HH:MM'")
        return v
    
class DateModel(BaseModel):
    date: str = Field(description="Properly formatted date", pattern=r'^\d{2}-\d{2}-\d{4}$')
    @field_validator("date")
    def check_format_date(cls, v):
        if not re.match(r'^\d{2}-\d{2}-\d{4}$', v):  # Ensures DD-MM-YYYY format
            raise ValueError("The date must be in the format 'DD-MM-YYYY'")
        return v
     
class IdentificationNumberModel(BaseModel):
    id: int = Field(description="Identification number (7 or 8 digits long)")
    @field_validator("id")
    def check_format_id(cls, v):
        if not re.match(r'^\d{7,8}$', str(v)):  # Convert to string before matching
            raise ValueError("The ID number should be a 7 or 8-digit number")
        return v


class ToolResultModel(BaseModel):
    action: str
    status: Literal["success", "unavailable", "no_availability", "not_found", "needs_input", "info", "out_of_scope"]
    message: str
    doctor_name: Optional[str] = None
    specialization: Optional[str] = None
    department: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    slot: Optional[str] = None
    old_slot: Optional[str] = None
    new_slot: Optional[str] = None
    patient_id: Optional[int] = None
    slots: Optional[List[str]] = None
    doctors: Optional[List[Dict[str, Any]]] = None
    alternatives: Optional[List[str]] = None
    recommended_specializations: Optional[List[str]] = None
