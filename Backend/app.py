from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import json
import numpy as np
import pandas as pd

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load models
clf = joblib.load('model1_classifier.pkl')
reg = joblib.load('model2_regressor.pkl')
scaler = joblib.load('scaler.pkl')

with open('feature_columns.json') as f:
    feature_columns = json.load(f)

# Columns that need scaling
scale_cols = [
    'age', 'bmi', 'cycle_length', 'stress_level', 'sleep_quality',
    'pain_level', 'mood_score', 'hydration_level', 'prev_cycle_gap',
    'consecutive_spotting_months', 'rolling_avg_stress_3m',
    'cycle_length_deviation', 'gap_variability', 'personal_spotting_rate',
    'personal_base_gap', 'month_number'
]

class CycleInput(BaseModel):
    age: float
    bmi: float
    cycle_length: float
    stress_level: float
    sleep_quality: float
    pain_level: float
    mood_score: float
    hydration_level: float
    prev_cycle_gap: float
    consecutive_spotting_months: float
    rolling_avg_stress_3m: float
    cycle_length_deviation: float
    gap_variability: float
    personal_spotting_rate: float
    personal_base_gap: float
    month_number: float
    travel_disruption: int
    recent_illness: int
    prev_cycle_had_spotting: int
    two_months_ago_spotting: int
    activity_level: int
    stress_trend: int
    bmi_category: int
    age_group: int
    flow_intensity: int
    season_autumn: int
    season_monsoon: int
    season_summer: int
    season_winter: int
    condition_PCOS: int
    condition_endometriosis: int
    condition_none: int
    condition_thyroid: int
    contraceptive_IUD: int
    contraceptive_implant: int
    contraceptive_none: int
    contraceptive_pill: int

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/predict")
def predict(data: CycleInput):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_columns]

    # Scale numeric columns
    input_df[scale_cols] = scaler.transform(input_df[scale_cols])

    # Model 1 — spotting prediction
    spotting_prob = clf.predict_proba(input_df)[0][1]
    spotting_pred = int(clf.predict(input_df)[0])

    # Caution zone — if probability between 40% and 60%
    if 0.40 <= spotting_prob <= 0.60:
        recommendation = (
            f"Uncertain ({round(spotting_prob*100)}% probability). "
            f"Cycle could go either way — carry a pad to be safe."
        )
        gap_days = None
        safe_window = 1

    elif spotting_pred == 1:
        gap_pred = float(reg.predict(input_df)[0])
        gap_days = round(gap_pred)
        safe_window = max(1, gap_days - 1)
        recommendation = (
            f"Spotting likely ({round(spotting_prob*100)}% probability). "
            f"Bleeding expected in {gap_days} days. "
            f"Carry a pad within {safe_window} day(s) of spotting."
        )
    else:
        gap_days = None
        safe_window = None
        recommendation = (
            f"Spotting unlikely ({round(spotting_prob*100)}% probability). "
            f"Period may start directly."
        )

    return {
        "spotting_probability": round(spotting_prob, 3),
        "spotting_predicted": spotting_pred,
        "gap_days_predicted": gap_days,
        "safe_window_days": safe_window,
        "recommendation": recommendation
    }