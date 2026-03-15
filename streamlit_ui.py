import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from app_service import execute_query, register_new_patient


st.set_page_config(page_title="Doctor Appointment System", page_icon=":hospital:", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(30, 64, 175, 0.18), transparent 24%),
            #0b1020;
        color: #f8fafc;
    }
    .block-container {
        max-width: 980px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(17, 24, 39, 0.76));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 24px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 24px 80px rgba(15, 23, 42, 0.28);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.04em;
        margin-bottom: 0.35rem;
        color: #f8fafc;
    }
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        margin-bottom: 0;
    }
    .status-chip {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(14, 165, 233, 0.14);
        color: #bae6fd;
        font-size: 0.9rem;
        margin-top: 0.75rem;
    }
    .panel-card {
        background: rgba(15, 23, 42, 0.76);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }
    .info-banner {
        background: rgba(8, 47, 73, 0.72);
        border: 1px solid rgba(56, 189, 248, 0.22);
        color: #dbeafe;
        border-radius: 18px;
        padding: 0.9rem 1rem;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    .patient-meta {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin-top: 0.75rem;
    }
    .meta-pill {
        background: rgba(30, 41, 59, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 999px;
        color: #e2e8f0;
        font-size: 0.88rem;
        padding: 0.45rem 0.75rem;
    }
    div[data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 20px;
        padding: 0.35rem 0.4rem;
        margin-bottom: 0.8rem;
    }
    div[data-testid="stChatInput"] {
        position: sticky;
        bottom: 0.5rem;
    }
    .quick-note {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-top: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "patient_id" not in st.session_state:
    st.session_state.patient_id = ""
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""
if "patient_phone" not in st.session_state:
    st.session_state.patient_phone = ""
if "patient_registered" not in st.session_state:
    st.session_state.patient_registered = False
if "patient_mode" not in st.session_state:
    st.session_state.patient_mode = "Returning Patient"


def reset_chat():
    st.session_state.chat_history = []


def submit_query(query: str):
    if st.session_state.patient_registered and st.session_state.patient_id:
        result = execute_query(
            id_number=int(st.session_state.patient_id),
            message=query,
            is_new_patient=False,
        )
    elif st.session_state.patient_mode == "New Patient":
        result = execute_query(
            id_number=None,
            message=query,
            is_new_patient=True,
            full_name=st.session_state.patient_name,
            phone=st.session_state.patient_phone,
        )
    else:
        result = execute_query(
            id_number=int(st.session_state.patient_id) if str(st.session_state.patient_id).isdigit() else None,
            message=query,
            is_new_patient=False,
        )

    patient = result.get("patient", {}) or {}
    if patient:
        st.session_state.patient_id = str(patient.get("patient_id", ""))
        st.session_state.patient_name = patient.get("full_name", st.session_state.patient_name)
        st.session_state.patient_phone = patient.get("phone", st.session_state.patient_phone)
        st.session_state.patient_registered = True
        st.session_state.patient_mode = "Returning Patient"

    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.chat_history.append({"role": "assistant", "content": result.get("reply", "")})


def proceed_patient_setup():
    if st.session_state.patient_mode == "Returning Patient":
        if not str(st.session_state.patient_id).isdigit():
            st.error("Enter a valid hospital ID to continue.")
            return
        st.session_state.patient_registered = True
        return

    if not st.session_state.patient_name.strip() or not st.session_state.patient_phone.strip():
        st.error("Enter your full name and phone number to continue.")
        return

    try:
        patient = register_new_patient(
            st.session_state.patient_name,
            st.session_state.patient_phone,
        )
        st.session_state.patient_id = str(patient.get("patient_id", ""))
        st.session_state.patient_name = patient.get("full_name", st.session_state.patient_name)
        st.session_state.patient_phone = patient.get("phone", st.session_state.patient_phone)
        st.session_state.patient_registered = True
        st.session_state.patient_mode = "Returning Patient"
        st.success(f"Registration complete. Your hospital ID is {st.session_state.patient_id}.")
    except Exception as exc:
        st.error(str(exc))


with st.sidebar:
    st.markdown("### Patient")
    st.caption("Use your hospital ID if you already registered. New patients can register once and continue chatting.")

    if not st.session_state.patient_registered:
        st.session_state.patient_mode = st.radio(
            "Patient Type",
            ["Returning Patient", "New Patient"],
            index=0 if st.session_state.patient_mode == "Returning Patient" else 1,
        )

        if st.session_state.patient_mode == "Returning Patient":
            st.session_state.patient_id = st.text_input("Hospital ID", st.session_state.patient_id)
        else:
            st.session_state.patient_name = st.text_input("Full name", st.session_state.patient_name)
            st.session_state.patient_phone = st.text_input("Phone number", st.session_state.patient_phone)
            st.markdown(
                "<div class='quick-note'>After your first message, your new hospital ID will be created automatically and this registration form will disappear.</div>",
                unsafe_allow_html=True,
            )

        st.button("Proceed", use_container_width=True, on_click=proceed_patient_setup, type="primary")
    else:
        st.success("Patient verified")
        st.markdown(f"**Name:** {st.session_state.patient_name}")
        st.markdown(f"**Hospital ID:** {st.session_state.patient_id}")
        if st.session_state.patient_phone:
            st.markdown(f"**Phone:** {st.session_state.patient_phone}")

    st.button("Clear Chat", use_container_width=True, on_click=reset_chat)


st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-title">Doctor Appointment System</div>
        <p class="hero-subtitle">Describe symptoms, ask for the right specialist, check schedules, or manage appointments in one chat.</p>
        <div class="status-chip">Chat-first patient assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-banner">
        Demo schedule data is currently available from <strong>05-08-2024</strong> to <strong>03-09-2024</strong>.
        Availability checks, booking, rescheduling, and cancellation are supported within this date range.
    </div>
    """,
    unsafe_allow_html=True,
)


if st.session_state.patient_registered:
    st.markdown(
        f"""
        <div class="panel-card">
            <strong>Signed in as {st.session_state.patient_name}</strong>
            <div class="patient-meta">
                <span class="meta-pill">Hospital ID: {st.session_state.patient_id}</span>
                <span class="meta-pill">Ready for booking, triage, and follow-up questions</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="panel-card">
            <strong>How this works</strong>
            <div class="quick-note">Returning patients can continue with their hospital ID. New patients only need name and phone once, then the system keeps them signed in for the session.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.markdown(
                "Tell me what you need. For example: `I have high fever`, `book a cardiologist for tomorrow`, or `cancel my appointment with Dr. John Doe`."
            )

    for message in st.session_state.chat_history:
        avatar_role = "assistant" if message["role"] == "assistant" else "user"
        with st.chat_message(avatar_role):
            st.markdown(message["content"])


prompt = st.chat_input("Message the hospital assistant")
if prompt:
    if not st.session_state.patient_registered and st.session_state.patient_mode == "Returning Patient":
        if not str(st.session_state.patient_id).isdigit():
            st.error("Enter a valid hospital ID before sending your message.")
        else:
            try:
                submit_query(prompt)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    elif not st.session_state.patient_registered and st.session_state.patient_mode == "New Patient":
        if not st.session_state.patient_name.strip() or not st.session_state.patient_phone.strip():
            st.error("Enter your name and phone number first.")
        else:
            try:
                submit_query(prompt)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    else:
        try:
            submit_query(prompt)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
