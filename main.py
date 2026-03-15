import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app_service import execute_query, register_new_patient


os.environ.pop("SSL_CERT_FILE", None)

app = FastAPI()


class UserQuery(BaseModel):
    id_number: Optional[int] = None
    messages: str
    is_new_patient: bool = False
    full_name: Optional[str] = None
    phone: Optional[str] = None


class PatientRegistration(BaseModel):
    full_name: str
    phone: str


@app.post("/patients/register")
def register_patient_endpoint(payload: PatientRegistration):
    try:
        patient = register_new_patient(payload.full_name, payload.phone)
        return {"patient": patient}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/execute")
def execute_agent(user_input: UserQuery):
    try:
        return execute_query(
            id_number=user_input.id_number,
            message=user_input.messages,
            is_new_patient=user_input.is_new_patient,
            full_name=user_input.full_name,
            phone=user_input.phone,
        )
    except ValueError as exc:
        status_code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
