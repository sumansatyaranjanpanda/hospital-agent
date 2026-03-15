# Doctor Appointment System

Single-process Streamlit app for:
- doctor recommendations
- doctor availability checks
- appointment booking
- appointment rescheduling
- appointment cancellation

The app uses:
- Streamlit UI
- LangGraph / LangChain agent workflow
- CSV demo data for schedules and patients

## Local Run

Create environment and install:

```bash
conda create -p venv python=3.10 -y
conda activate ./venv
pip install -r requirements.txt
```

Set your API key in `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
```

Run the app:

```bash
streamlit run streamlit_ui.py
```

## Streamlit Community Cloud Deployment

This project now runs as a single Streamlit app. No separate FastAPI deployment is required.

### Streamlit Cloud settings

- Repository: your GitHub repo
- Branch: your deployment branch
- Main file path: `streamlit_ui.py`
- Python version: `3.10`

### Streamlit Cloud secrets

Add this in the app `Secrets` section:

```toml
OPENAI_API_KEY="your_openai_api_key"
```

The app automatically reads `OPENAI_API_KEY` from:
1. Streamlit secrets
2. `.env` locally

## Optional FastAPI Wrapper

If you still want an HTTP API, `main.py` remains available as a thin wrapper around the same shared service logic.

## Important Note

This project still stores bookings and patient registrations in CSV files:
- [data/doctor_availability.csv](/c:/Users/rajap/doctor-appoitment-multiagent/data/doctor_availability.csv)
- [data/patients.csv](/c:/Users/rajap/doctor-appoitment-multiagent/data/patients.csv)

That is fine for demo use, but changes may not persist across restarts on some cloud platforms.

For stronger persistence, move the data layer to SQLite or Postgres.
