# Doctor Appointment System

Hospital-style multi-agent appointment assistant built with:
- FastAPI backend
- Streamlit frontend
- LangGraph / LangChain orchestration
- CSV-based demo data for doctors, schedules, and patients

## Local Run

Create environment and install dependencies:

```bash
conda create -p venv python=3.10 -y
conda activate ./venv
pip install -r requirements.txt
```

Start backend:

```bash
uvicorn main:app --reload
```

Start frontend:

```bash
streamlit run streamlit_ui.py
```

## Deployment

Recommended setup:
- Streamlit frontend on Streamlit Community Cloud
- FastAPI backend on Render, Railway, Fly.io, or similar

### 1. Deploy FastAPI backend

This repo includes a [Procfile](/c:/Users/rajap/doctor-appoitment-multiagent/Procfile) with:

```bash
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Backend environment variables:

```bash
OPENAI_API_KEY=your_key_here
```

### 2. Deploy Streamlit frontend

The frontend reads the backend URL from `API_URL`:

```python
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
```

Set this in Streamlit Cloud secrets:

```toml
API_URL="https://your-backend-url"
```

Then deploy `streamlit_ui.py` as the Streamlit entry file.

### 3. Streamlit Cloud steps

1. Push this repo to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from the repo.
4. Select:
   - Branch: your deployment branch
   - Main file path: `streamlit_ui.py`
5. Add secret:

```toml
API_URL="https://your-backend-url"
```

## Important Note

This project currently stores bookings and patient registrations in CSV files:
- [data/doctor_availability.csv](/c:/Users/rajap/doctor-appoitment-multiagent/data/doctor_availability.csv)
- [data/patients.csv](/c:/Users/rajap/doctor-appoitment-multiagent/data/patients.csv)

That is fine for demo use, but on many cloud platforms file changes may not persist across restarts or redeploys.

For a more reliable deployment, move these to:
- SQLite for simple persistence
- Postgres for production
