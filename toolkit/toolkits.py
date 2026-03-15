from datetime import datetime
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from data_models.models import DateModel, DateTimeModel, IdentificationNumberModel


DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "doctor_availability.csv"
CATALOG_FILE = Path(__file__).resolve().parent.parent / "data" / "doctor_catalog.csv"

TRIAGE_RULES = {
    "high fever": ["general_physician", "pulmonologist"],
    "fever": ["general_physician"],
    "cold": ["general_physician", "ent"],
    "cough": ["general_physician", "pulmonologist"],
    "breathing": ["pulmonologist"],
    "chest pain": ["cardiologist"],
    "heart": ["cardiologist"],
    "skin": ["dermatologist"],
    "rash": ["dermatologist"],
    "tooth": ["general_dentist"],
    "teeth": ["general_dentist"],
    "braces": ["orthodontist"],
    "jaw": ["oral_surgeon"],
    "child": ["pediatrician"],
    "baby": ["pediatrician"],
    "ear": ["ent"],
    "nose": ["ent"],
    "throat": ["ent"],
    "eye": ["ophthalmologist"],
    "vision": ["ophthalmologist"],
    "stomach": ["gastroenterologist"],
    "diabetes": ["endocrinologist"],
    "hormone": ["endocrinologist"],
    "urine": ["urologist"],
    "kidney": ["urologist"],
    "stress": ["psychiatrist"],
    "anxiety": ["psychiatrist"],
    "back pain": ["orthopedic"],
    "bone": ["orthopedic"],
    "period": ["gynecologist"],
    "pregnancy": ["gynecologist"],
    "headache": ["general_physician", "neurologist"],
    "migraine": ["neurologist"],
}


def _load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["date_slot"] = pd.to_datetime(df["date_slot"], format="%d-%m-%Y %H:%M", errors="coerce")
    return df


def _load_catalog() -> pd.DataFrame:
    return pd.read_csv(CATALOG_FILE)


def _save_df(df: pd.DataFrame) -> None:
    output = df.copy()
    output["date_slot"] = output["date_slot"].dt.strftime("%d-%m-%Y %H:%M")
    output.to_csv(DATA_FILE, index=False)


def _format_slot(value: datetime) -> str:
    return value.strftime("%d-%m-%Y %H:%M")


def _friendly_time(value: datetime) -> str:
    return value.strftime("%I:%M %p").lstrip("0")


def _friendly_slot(value: datetime) -> str:
    return f"{value.strftime('%d-%m-%Y')} at {_friendly_time(value)}"


def _status_payload(action: str, status: str, message: str, **details):
    return {"action": action, "status": status, "message": message, **details}


def _find_catalog_matches(specialization: str) -> pd.DataFrame:
    catalog = _load_catalog()
    return catalog[catalog["specialization"].str.lower() == specialization.lower()].copy()


def _same_day_available_slots(df: pd.DataFrame, doctor_name: str, target_slot: datetime, limit: int = 3) -> list[str]:
    same_day = df[
        (df["doctor_name"].str.lower() == doctor_name.lower())
        & (df["is_available"] == True)
        & (df["date_slot"].dt.date == target_slot.date())
    ].sort_values("date_slot")
    return [_friendly_time(slot) for slot in same_day["date_slot"].head(limit)]


@tool
def find_doctors_by_specialization(specialization: str):
    """
    Return doctors registered under a hospital specialization or department.
    """
    catalog = _load_catalog()
    query = specialization.strip().lower().replace(" ", "_")
    rows = catalog[
        (catalog["specialization"].str.lower() == query)
        | (catalog["department"].str.lower() == specialization.strip().lower())
        | (catalog["specialization"].str.lower().str.contains(specialization.strip().lower(), na=False))
    ]

    if rows.empty:
        return _status_payload(
            "find_doctors_by_specialization",
            "not_found",
            f"No doctors were found for specialization '{specialization}'.",
            specialization=query,
            doctors=[],
        )

    doctors = rows[["doctor_name", "specialization", "department"]].to_dict(orient="records")
    return _status_payload(
        "find_doctors_by_specialization",
        "success",
        f"Found {len(doctors)} doctor(s) for {specialization}.",
        specialization=query,
        doctors=doctors,
    )


@tool
def recommend_doctor_for_query(query: str):
    """
    Recommend the right doctor specialization based on patient symptoms or concerns.
    """
    lowered = query.lower()
    matches = []
    for keyword, specializations in TRIAGE_RULES.items():
        if keyword in lowered:
            matches.extend(specializations)

    if not matches:
        return _status_payload(
            "recommend_doctor_for_query",
            "needs_input",
            "I need a little more detail about the symptom or concern to suggest the right doctor.",
            recommended_specializations=[],
        )

    ordered = []
    for specialization in matches:
        if specialization not in ordered:
            ordered.append(specialization)

    catalog = _load_catalog()
    doctors = catalog[catalog["specialization"].isin(ordered)][
        ["doctor_name", "specialization", "department"]
    ].to_dict(orient="records")

    return _status_payload(
        "recommend_doctor_for_query",
        "info",
        f"Recommended doctor type(s) for the query: {', '.join(ordered)}.",
        recommended_specializations=ordered,
        doctors=doctors,
    )


@tool
def check_availability_by_doctor(desired_date: DateModel, doctor_name: str):
    """
    Check available slots for a specific doctor on a given date.
    """
    df = _load_df()
    target_date = datetime.strptime(desired_date.date, "%d-%m-%Y").date()
    rows = df[
        (df["date_slot"].dt.date == target_date)
        & (df["doctor_name"].str.lower() == doctor_name.lower())
        & (df["is_available"] == True)
    ].sort_values("date_slot")

    slots = [_friendly_time(slot) for slot in rows["date_slot"].tolist()]
    if not slots:
        return _status_payload(
            "check_availability_by_doctor",
            "no_availability",
            f"No free slots were found for Dr. {doctor_name.title()} on {desired_date.date}.",
            doctor_name=doctor_name,
            date=desired_date.date,
            slots=[],
        )

    return _status_payload(
        "check_availability_by_doctor",
        "success",
        f"Found {len(slots)} available slot(s) for Dr. {doctor_name.title()} on {desired_date.date}.",
        doctor_name=doctor_name,
        date=desired_date.date,
        slots=slots,
    )


@tool
def check_availability_by_specialization(desired_date: DateModel, specialization: str):
    """
    Check available slots for all doctors in a specialization on a given date.
    """
    df = _load_df()
    target_date = datetime.strptime(desired_date.date, "%d-%m-%Y").date()
    normalized = specialization.strip().lower().replace(" ", "_")
    rows = df[
        (df["date_slot"].dt.date == target_date)
        & (df["specialization"].str.lower() == normalized)
        & (df["is_available"] == True)
    ].sort_values("date_slot")

    if rows.empty:
        catalog_matches = _find_catalog_matches(normalized)
        return _status_payload(
            "check_availability_by_specialization",
            "no_availability",
            f"No free slots were found for {specialization} on {desired_date.date}.",
            specialization=normalized,
            date=desired_date.date,
            doctors=catalog_matches[["doctor_name", "specialization", "department"]].to_dict(orient="records"),
        )

    doctor_groups = []
    for doctor_name, group in rows.groupby("doctor_name"):
        doctor_groups.append(
            {
                "doctor_name": doctor_name,
                "slots": [_friendly_time(slot) for slot in group["date_slot"].tolist()],
            }
        )

    return _status_payload(
        "check_availability_by_specialization",
        "success",
        f"Found availability for {specialization} on {desired_date.date}.",
        specialization=normalized,
        date=desired_date.date,
        doctors=doctor_groups,
    )


@tool
def list_all_specializations():
    """
    List all doctor types available in the hospital catalog.
    """
    catalog = _load_catalog()
    specializations = sorted(catalog["specialization"].dropna().unique().tolist())
    return _status_payload(
        "list_all_specializations",
        "success",
        f"Found {len(specializations)} specializations in the hospital catalog.",
        recommended_specializations=specializations,
    )


@tool
def set_appointment(desired_date: DateTimeModel, id_number: IdentificationNumberModel, doctor_name: str):
    """
    Book a specific doctor slot for a patient.
    """
    df = _load_df()
    target_slot = datetime.strptime(desired_date.date, "%d-%m-%Y %H:%M")
    mask = (
        (df["date_slot"] == target_slot)
        & (df["doctor_name"].str.lower() == doctor_name.lower())
        & (df["is_available"] == True)
    )

    if not mask.any():
        alternatives = _same_day_available_slots(df, doctor_name, target_slot)
        return _status_payload(
            "set_appointment",
            "unavailable",
            f"The requested slot with Dr. {doctor_name.title()} is not available.",
            doctor_name=doctor_name,
            date=target_slot.strftime("%d-%m-%Y"),
            time=_friendly_time(target_slot),
            patient_id=id_number.id,
            alternatives=alternatives,
        )

    df.loc[mask, ["is_available", "patient_to_attend"]] = [False, id_number.id]
    _save_df(df)
    return _status_payload(
        "set_appointment",
        "success",
        f"Appointment booked with Dr. {doctor_name.title()} for {_friendly_slot(target_slot)}.",
        doctor_name=doctor_name,
        slot=_format_slot(target_slot),
        patient_id=id_number.id,
    )


@tool
def cancel_appointment(date: DateTimeModel, id_number: IdentificationNumberModel, doctor_name: str):
    """
    Cancel an existing appointment for a patient.
    """
    df = _load_df()
    target_slot = datetime.strptime(date.date, "%d-%m-%Y %H:%M")
    mask = (
        (df["date_slot"] == target_slot)
        & (df["patient_to_attend"].fillna(0).astype(int) == id_number.id)
        & (df["doctor_name"].str.lower() == doctor_name.lower())
    )

    if not mask.any():
        return _status_payload(
            "cancel_appointment",
            "not_found",
            "No matching appointment was found to cancel.",
            doctor_name=doctor_name,
            slot=_format_slot(target_slot),
            patient_id=id_number.id,
        )

    df.loc[mask, ["is_available", "patient_to_attend"]] = [True, None]
    _save_df(df)
    return _status_payload(
        "cancel_appointment",
        "success",
        f"Appointment with Dr. {doctor_name.title()} on {_friendly_slot(target_slot)} was cancelled.",
        doctor_name=doctor_name,
        slot=_format_slot(target_slot),
        patient_id=id_number.id,
    )


@tool
def reschedule_appointment(
    old_date: DateTimeModel,
    new_date: DateTimeModel,
    id_number: IdentificationNumberModel,
    doctor_name: str,
):
    """
    Move an existing appointment to a new slot with the same doctor.
    """
    df = _load_df()
    old_slot = datetime.strptime(old_date.date, "%d-%m-%Y %H:%M")
    new_slot = datetime.strptime(new_date.date, "%d-%m-%Y %H:%M")

    current_mask = (
        (df["date_slot"] == old_slot)
        & (df["patient_to_attend"].fillna(0).astype(int) == id_number.id)
        & (df["doctor_name"].str.lower() == doctor_name.lower())
    )
    if not current_mask.any():
        return _status_payload(
            "reschedule_appointment",
            "not_found",
            "No existing appointment was found for the original slot.",
            doctor_name=doctor_name,
            old_slot=_format_slot(old_slot),
            new_slot=_format_slot(new_slot),
            patient_id=id_number.id,
        )

    new_mask = (
        (df["date_slot"] == new_slot)
        & (df["doctor_name"].str.lower() == doctor_name.lower())
        & (df["is_available"] == True)
    )
    if not new_mask.any():
        alternatives = _same_day_available_slots(df, doctor_name, new_slot)
        return _status_payload(
            "reschedule_appointment",
            "unavailable",
            f"The new slot with Dr. {doctor_name.title()} is not available.",
            doctor_name=doctor_name,
            old_slot=_format_slot(old_slot),
            new_slot=_format_slot(new_slot),
            patient_id=id_number.id,
            alternatives=alternatives,
        )

    df.loc[current_mask, ["is_available", "patient_to_attend"]] = [True, None]
    df.loc[new_mask, ["is_available", "patient_to_attend"]] = [False, id_number.id]
    _save_df(df)
    return _status_payload(
        "reschedule_appointment",
        "success",
        f"Appointment moved from {_friendly_slot(old_slot)} to {_friendly_slot(new_slot)} with Dr. {doctor_name.title()}.",
        doctor_name=doctor_name,
        old_slot=_format_slot(old_slot),
        new_slot=_format_slot(new_slot),
        patient_id=id_number.id,
    )
