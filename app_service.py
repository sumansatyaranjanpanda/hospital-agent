import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent import DoctorAppointmentAgent
from data_models.models import DateModel, DateTimeModel, IdentificationNumberModel
from toolkit.toolkits import (
    cancel_appointment,
    check_availability_by_doctor,
    reschedule_appointment,
    set_appointment,
)
from utils.patients import get_patient, register_patient


agent = DoctorAppointmentAgent()
SESSION_STORE: Dict[int, List[BaseMessage]] = {}
SESSION_CONTEXT: Dict[int, Dict[str, str]] = {}
DATA_FILE = Path(__file__).resolve().parent / "data" / "doctor_availability.csv"

DATE_PATTERNS = [
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%Y-%m-%d",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%b %d %Y",
]


def _parse_flexible_date(raw_date: str, fallback_year: Optional[int] = None) -> Optional[str]:
    cleaned = raw_date.strip().replace(",", "")
    candidate_formats = DATE_PATTERNS + ["%d %B", "%d %b", "%B %d", "%b %d"]

    for fmt in candidate_formats:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            if "%Y" not in fmt:
                parsed = parsed.replace(year=fallback_year or datetime.now().year)
            return parsed.strftime("%d-%m-%Y")
        except ValueError:
            continue
    return None


def _parse_flexible_time(raw_time: str) -> Optional[str]:
    cleaned = raw_time.strip().upper().replace(".", ":")
    patterns = ["%I:%M %p", "%I %p", "%H:%M", "%H"]
    for fmt in patterns:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            return parsed.strftime("%H:%M")
        except ValueError:
            continue
    return None


def _extract_any_date_from_text(text: str, fallback_year: Optional[int] = None) -> Optional[str]:
    patterns = [
        r"\d{2}-\d{2}-\d{4}",
        r"\d{2}/\d{2}/\d{4}",
        r"\d{4}-\d{2}-\d{2}",
        r"[A-Z][a-z]+ \d{1,2}, \d{4}",
        r"\d{1,2} [A-Z][a-z]+ \d{4}",
        r"[A-Z][a-z]{2,8} \d{1,2}",
        r"\d{1,2} [A-Z][a-z]{2,8}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            parsed = _parse_flexible_date(match.group(0), fallback_year=fallback_year)
            if parsed:
                return parsed
    return None


def _extract_doctor_and_date(history: List[BaseMessage]):
    doctor_name = None
    date_value = None
    doctor_pattern = re.compile(r"Dr\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)")
    for message in reversed(history):
        content = getattr(message, "content", "")
        if not doctor_name:
            doctor_match = doctor_pattern.search(content)
            if doctor_match:
                doctor_name = doctor_match.group(1).lower()
        if not date_value:
            fallback_year = None
            explicit_year = re.search(r"(20\d{2})", content)
            if explicit_year:
                fallback_year = int(explicit_year.group(1))
            date_value = _extract_any_date_from_text(content, fallback_year=fallback_year)
        if doctor_name and date_value:
            break

    return doctor_name, date_value


def _get_context(patient_id: int) -> Dict[str, str]:
    return SESSION_CONTEXT.setdefault(patient_id, {})


def _save_context(patient_id: int, **values):
    context = _get_context(patient_id)
    for key, value in values.items():
        if value is None:
            context.pop(key, None)
        else:
            context[key] = value


def _find_patient_appointment(patient_id: int, doctor_name: Optional[str] = None):
    rows = []
    with DATA_FILE.open() as handle:
        for row in csv.DictReader(handle):
            attendee = str(row.get("patient_to_attend", "")).split(".")[0]
            if attendee == str(patient_id):
                if doctor_name and row["doctor_name"].lower() != doctor_name.lower():
                    continue
                rows.append(row)
    rows.sort(key=lambda item: datetime.strptime(item["date_slot"], "%d-%m-%Y %H:%M"))
    return rows[-1] if rows else None


def _format_availability_reply(result: dict, doctor_name: str, date_value: str) -> str:
    if result.get("status") == "success":
        slots = result.get("slots", [])
        if slots:
            return (
                f"Dr. {doctor_name.title()} is available on {date_value}. "
                f"Open slots: {', '.join(slots)}. Let me know which time you want to book."
            )
    return f"Dr. {doctor_name.title()} does not have any free slots on {date_value}. Would you like another date?"


def _format_booking_reply(result: dict, doctor_name: str, date_value: str, time_value: str) -> str:
    if result.get("status") == "success":
        return (
            f"Your appointment with Dr. {doctor_name.title()} is confirmed for {date_value} at {time_value}. "
            "If you need anything else, I can help with that too."
        )

    alternatives = result.get("alternatives", []) or []
    if alternatives:
        return (
            f"The {time_value} slot with Dr. {doctor_name.title()} on {date_value} is not available. "
            f"Available options are: {', '.join(alternatives)}. Tell me which one you want to book."
        )

    return f"Dr. {doctor_name.title()} does not have a free slot at {time_value} on {date_value}. Would you like another date?"


def _format_reschedule_reply(result: dict, doctor_name: str, old_slot: str, new_slot: str) -> str:
    if result.get("status") == "success":
        return f"Your appointment with Dr. {doctor_name.title()} has been moved from {old_slot} to {new_slot}."
    alternatives = result.get("alternatives", []) or []
    if alternatives:
        return (
            f"The requested new time with Dr. {doctor_name.title()} is not available. "
            f"Available options are: {', '.join(alternatives)}."
        )
    return f"I could not reschedule your appointment with Dr. {doctor_name.title()}. Please try another time."


def _format_cancel_reply(result: dict, doctor_name: str, slot: str) -> str:
    if result.get("status") == "success":
        return f"Your appointment with Dr. {doctor_name.title()} on {slot} has been cancelled."
    return f"I could not find an appointment with Dr. {doctor_name.title()} on {slot} to cancel."


def _handle_direct_follow_up(message: str, history: List[BaseMessage], patient_id: int):
    cleaned = message.strip()
    if not cleaned:
        return None

    context = _get_context(patient_id)
    doctor_name, date_value = _extract_doctor_and_date(history)
    doctor_name = doctor_name or context.get("last_doctor")
    date_value = date_value or context.get("last_date")
    fallback_year = int(date_value[-4:]) if date_value else None
    embedded_date = _extract_any_date_from_text(cleaned, fallback_year=fallback_year)
    lowered = cleaned.lower()
    availability_phrases = [
        "free slot",
        "available slot",
        "availability",
        "check slot",
        "check availability",
        "slot available",
        "show slots",
        "available time",
        "free time",
    ]

    if doctor_name and (embedded_date or any(phrase in lowered for phrase in availability_phrases)):
        target_date = embedded_date or date_value
        if target_date:
            tool_result = check_availability_by_doctor.invoke(
                {"desired_date": DateModel(date=target_date), "doctor_name": doctor_name}
            )
            _save_context(patient_id, last_doctor=doctor_name, last_date=target_date)
            return _format_availability_reply(tool_result, doctor_name, target_date)

    time_match = re.search(r"\b(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\b", cleaned)
    booking_phrases = ["book", "confirm", "take this", "book on this", "schedule"]
    if doctor_name and date_value and time_match and (
        any(phrase in lowered for phrase in booking_phrases) or re.fullmatch(r"\d{1,2}:\d{2}\s*(AM|PM|am|pm)", cleaned)
    ):
        parsed_time = _parse_flexible_time(time_match.group(1))
        if parsed_time:
            tool_result = set_appointment.invoke(
                {
                    "desired_date": DateTimeModel(date=f"{date_value} {parsed_time}"),
                    "id_number": IdentificationNumberModel(id=patient_id),
                    "doctor_name": doctor_name,
                }
            )
            friendly_time = datetime.strptime(parsed_time, "%H:%M").strftime("%I:%M %p").lstrip("0")
            if tool_result.get("status") == "success":
                _save_context(
                    patient_id,
                    last_doctor=doctor_name,
                    last_date=date_value,
                    last_time=parsed_time,
                    pending_action=None,
                    pending_new_date=None,
                    pending_new_time=None,
                )
            return _format_booking_reply(tool_result, doctor_name, date_value, friendly_time)

    existing_appt = _find_patient_appointment(patient_id, doctor_name=doctor_name) if doctor_name else _find_patient_appointment(patient_id)
    if existing_appt:
        existing_doctor = existing_appt["doctor_name"].lower()
        existing_slot = datetime.strptime(existing_appt["date_slot"], "%d-%m-%Y %H:%M")
        existing_date = existing_slot.strftime("%d-%m-%Y")
        existing_time = existing_slot.strftime("%H:%M")
        _save_context(patient_id, last_doctor=existing_doctor, last_date=existing_date, last_time=existing_time)

        if any(phrase in lowered for phrase in ["reschedule", "move appointment", "change appointment", "update appointment", "update my appointment", "i need to update", "update it"]):
            _save_context(patient_id, pending_action="reschedule")
            return (
                f"Please share the new date and time for your appointment with Dr. {existing_doctor.title()}. "
                f"Your current booking is on {existing_date} at {existing_slot.strftime('%I:%M %p').lstrip('0')}."
            )

        if any(phrase in lowered for phrase in ["delete my appointment", "cancel appointment", "cancel my appointment", "delete appointment", "delete my apporintment", "delet", "cancel it", "remove appointment"]):
            _save_context(patient_id, pending_action="cancel")
            return (
                f"Please confirm that you want to cancel your appointment with Dr. {existing_doctor.title()} "
                f"on {existing_date} at {existing_slot.strftime('%I:%M %p').lstrip('0')}."
            )

        if _get_context(patient_id).get("pending_action") == "cancel" and (
            lowered in {"yes", "yes please", "confirm", "yes confirm"} or "delete" in lowered or "cancel" in lowered
        ):
            tool_result = cancel_appointment.invoke(
                {
                    "date": DateTimeModel(date=f"{existing_date} {existing_time}"),
                    "id_number": IdentificationNumberModel(id=patient_id),
                    "doctor_name": existing_doctor,
                }
            )
            _save_context(patient_id, pending_action=None)
            return _format_cancel_reply(
                tool_result,
                existing_doctor,
                f"{existing_date} at {existing_slot.strftime('%I:%M %p').lstrip('0')}",
            )

        pending_action = context.get("pending_action")
        if pending_action == "reschedule_confirm" and lowered in {"yes", "yes please", "confirm", "yes confirm"}:
            new_date = context.get("pending_new_date")
            new_time = context.get("pending_new_time")
            if new_date and new_time:
                tool_result = reschedule_appointment.invoke(
                    {
                        "old_date": DateTimeModel(date=f"{existing_date} {existing_time}"),
                        "new_date": DateTimeModel(date=f"{new_date} {new_time}"),
                        "id_number": IdentificationNumberModel(id=patient_id),
                        "doctor_name": existing_doctor,
                    }
                )
                _save_context(
                    patient_id,
                    pending_action=None,
                    pending_new_date=None,
                    pending_new_time=None,
                    last_date=new_date if tool_result.get("status") == "success" else existing_date,
                    last_time=new_time if tool_result.get("status") == "success" else existing_time,
                )
                return _format_reschedule_reply(
                    tool_result,
                    existing_doctor,
                    f"{existing_date} at {existing_slot.strftime('%I:%M %p').lstrip('0')}",
                    f"{new_date} at {datetime.strptime(new_time, '%H:%M').strftime('%I:%M %p').lstrip('0')}",
                )

        if pending_action == "reschedule":
            same_day_markers = ["same day", "that same day", "same date", "on that day", "on same day", "to "]
            new_date = embedded_date or (existing_date if any(marker in lowered for marker in same_day_markers) else None)
            if not new_date:
                new_date = _extract_any_date_from_text(cleaned, fallback_year=int(existing_date[-4:]))
            new_time = _parse_flexible_time(time_match.group(1)) if time_match else None

            if new_date and new_time:
                _save_context(patient_id, pending_action="reschedule_confirm", pending_new_date=new_date, pending_new_time=new_time)
                return (
                    f"Please confirm that you want to move your appointment with Dr. {existing_doctor.title()} "
                    f"from {existing_date} at {existing_slot.strftime('%I:%M %p').lstrip('0')} to {new_date} at "
                    f"{datetime.strptime(new_time, '%H:%M').strftime('%I:%M %p').lstrip('0')}."
                )

            if new_date or new_time:
                display_date = new_date or existing_date
                display_time = datetime.strptime((new_time or existing_time), "%H:%M").strftime("%I:%M %p").lstrip("0")
                _save_context(
                    patient_id,
                    pending_action="reschedule_confirm",
                    pending_new_date=display_date,
                    pending_new_time=new_time or existing_time,
                )
                return (
                    f"Please confirm that you want to reschedule your appointment with Dr. {existing_doctor.title()} "
                    f"to {display_date} at {display_time}."
                )

    return None


def _fallback_reply(message: str) -> str:
    lowered = message.strip().lower()
    if any(word in lowered for word in ["hi", "hello", "hey", "hii", "hrllo"]):
        return "Hello. I can help you find the right doctor, check availability, book appointments, or manage an existing booking."
    if "any one there" in lowered or "anyone there" in lowered:
        return "I'm here. You can ask about symptoms, doctors, available slots, or appointment booking."
    return "I can help with doctor recommendations, appointment availability, booking, cancellation, and rescheduling. Tell me what you need."


def _normalize_follow_up_query(message: str, history: List[BaseMessage], patient_id: int) -> str:
    cleaned = message.strip()
    if not cleaned:
        return message

    doctor_name, date_value = _extract_doctor_and_date(history)
    lowered = cleaned.lower()

    time_only = re.fullmatch(r"(\d{1,2}:\d{2}\s*(AM|PM|am|pm))", cleaned)
    if time_only and doctor_name and date_value:
        parsed_time = datetime.strptime(time_only.group(1).upper(), "%I:%M %p").strftime("%H:%M")
        return (
            f"Please book an appointment with Dr. {doctor_name.title()} on {date_value} at {parsed_time} "
            f"for patient ID {patient_id}."
        )

    date_only = _parse_flexible_date(cleaned, fallback_year=int(date_value[-4:]) if date_value else None)
    if date_only:
        if doctor_name:
            return f"Please check all available slots for Dr. {doctor_name.title()} on {date_only} for patient ID {patient_id}."
        return f"Please use {date_only} as the appointment date and continue with the previous conversation."

    availability_phrases = [
        "free slot",
        "available slot",
        "availability",
        "check slot",
        "check availability",
        "slot available",
        "show slots",
    ]
    if any(phrase in lowered for phrase in availability_phrases):
        if doctor_name and date_value:
            return f"Please check all available slots for Dr. {doctor_name.title()} on {date_value}."
        if doctor_name:
            return f"Please ask for the date needed to check Dr. {doctor_name.title()}'s available slots."

    if re.fullmatch(r"\d{7,8}", cleaned):
        return f"My hospital ID is {cleaned}. Please continue with the appointment process."

    return message


def register_new_patient(full_name: str, phone: str) -> dict:
    return register_patient(full_name, phone)


def execute_query(
    *,
    id_number: Optional[int],
    message: str,
    is_new_patient: bool = False,
    full_name: Optional[str] = None,
    phone: Optional[str] = None,
):
    patient = None
    if is_new_patient:
        if not full_name or not phone:
            raise ValueError("New patients must provide full_name and phone.")
        patient = register_patient(full_name, phone)
        patient_id = int(patient["patient_id"])
    else:
        if id_number is None:
            raise ValueError("Returning patients must provide id_number.")
        patient = get_patient(int(id_number))
        if patient is None:
            raise ValueError("Patient ID not found. Please register as a new patient.")
        patient_id = int(patient["patient_id"])

    app_graph = agent.workflow()
    history = SESSION_STORE.get(patient_id, [])
    direct_reply = _handle_direct_follow_up(message, history, patient_id)
    if direct_reply:
        updated_messages = history + [HumanMessage(content=message), AIMessage(content=direct_reply)]
        SESSION_STORE[patient_id] = updated_messages[-12:]
        return {
            "reply": direct_reply,
            "patient": patient,
            "messages": [entry.content for entry in updated_messages],
        }

    normalized_query = _normalize_follow_up_query(message, history, patient_id)
    conversation = history + [HumanMessage(content=normalized_query)]

    query_data = {
        "messages": conversation,
        "id_number": patient_id,
        "next": "",
        "query": normalized_query,
        "current_reasoning": "",
        "worker_name": "",
        "worker_summary": "",
    }

    response = app_graph.invoke(query_data, config={"recursion_limit": 10})
    final_messages = response["messages"]
    SESSION_STORE[patient_id] = final_messages[-12:]

    final_reply = ""
    for item in reversed(final_messages):
        if isinstance(item, AIMessage):
            final_reply = item.content
            break

    if not final_reply.strip():
        final_reply = _fallback_reply(message)
        final_messages = final_messages + [AIMessage(content=final_reply)]
        SESSION_STORE[patient_id] = final_messages[-12:]

    return {
        "reply": final_reply,
        "patient": patient,
        "messages": [entry.content for entry in final_messages],
    }
