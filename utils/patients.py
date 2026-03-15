from pathlib import Path

import pandas as pd


PATIENTS_FILE = Path(__file__).resolve().parent.parent / "data" / "patients.csv"


def _load_patients() -> pd.DataFrame:
    if not PATIENTS_FILE.exists():
        return pd.DataFrame(columns=["patient_id", "full_name", "phone", "is_registered"])
    df = pd.read_csv(PATIENTS_FILE)
    if df.empty:
        return pd.DataFrame(columns=["patient_id", "full_name", "phone", "is_registered"])
    df["patient_id"] = df["patient_id"].astype(int)
    df["phone"] = df["phone"].astype(str)
    df["is_registered"] = df["is_registered"].astype(bool)
    return df


def _save_patients(df: pd.DataFrame) -> None:
    df.to_csv(PATIENTS_FILE, index=False)


def get_patient(patient_id: int):
    df = _load_patients()
    match = df[df["patient_id"] == int(patient_id)]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def register_patient(full_name: str, phone: str) -> dict:
    df = _load_patients()
    cleaned_phone = "".join(char for char in str(phone) if char.isdigit())
    if len(cleaned_phone) < 10:
        raise ValueError("Phone number must contain at least 10 digits.")

    existing = df[
        (df["full_name"].str.lower() == full_name.strip().lower())
        & (df["phone"].astype(str) == cleaned_phone)
    ]
    if not existing.empty:
        return existing.iloc[0].to_dict()

    next_id = 1000001 if df.empty else int(df["patient_id"].max()) + 1
    patient = {
        "patient_id": next_id,
        "full_name": full_name.strip(),
        "phone": cleaned_phone,
        "is_registered": True,
    }
    updated = pd.concat([df, pd.DataFrame([patient])], ignore_index=True)
    _save_patients(updated)
    return patient
