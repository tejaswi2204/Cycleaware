# CycleAware — Personal Cycle Prediction Tool

> A machine learning powered early warning system for menstrual spotting and bleeding patterns.

**Author:** S. Lakshmi Tejaswi

---

## Overview

Most period tracking applications predict when your next period will start. CycleAware goes deeper it predicts two things that no existing app addresses specifically:

1. **Will spotting occur this cycle?** (Yes / No)
2. **If spotting occurs, how many days before bleeding begins?**

This distinction matters in real life. Many people experience spotting before their period begins, with a gap of 1 to 5 days before bleeding starts. This gap varies month to month based on stress, lifestyle, health conditions, and personal history. CycleAware learns your pattern and gives you an early warning  so you are never caught off guard.

---

## The Problem

Spotting patterns are highly personal and inconsistent. Some months the gap is 3 days, some months it is 1 day, and occasionally a cycle skips spotting entirely and goes straight to bleeding. Existing period apps treat the period as a single event and do not model this internal structure.

CycleAware was built to solve exactly this gap  predicting not just when your period starts, but how it will start.

---

## How It Works

CycleAware uses two machine learning models working in sequence:

```
User inputs cycle details
        ↓
Model 1 (Random Forest Classifier)
Predicts: Will spotting occur? (Yes / No)
        ↓ if Yes
Model 2 (Random Forest Regressor)
Predicts: How many days until bleeding starts? (1 to 5)
        ↓
Recommendation displayed to user
```

The models were trained on synthetic data generated to reflect diverse real-world cycle patterns across different age groups, medical conditions, lifestyle factors, and cycle histories.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Machine Learning | Scikit-learn (Random Forest) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Data Generation | Python (NumPy, Pandas) |


---

## Features

- Predicts spotting probability for the current cycle
- Predicts the spotting-to-bleeding gap in days
- Provides a plain English recommendation with a safe window
- Mobile-friendly interface with 3 input screens
- Welcome page with full feature explanations including BMI calculator
- Season auto-detected from month number
- Handles uncertain predictions with a caution zone (40–60% probability)

---

## Project Structure

```
cycleaware/
├── backend/
│   ├── app.py                  — FastAPI backend with /predict and /health endpoints
│   ├── requirements.txt        — Python dependencies for deployment
│   ├── feature_columns.json    — Feature column names used during training
│   ├── model1_classifier.pkl   — Trained Random Forest Classifier (Model 1)
│   ├── model2_regressor.pkl    — Trained Random Forest Regressor (Model 2)
│   └── scaler.pkl              — StandardScaler fitted during preprocessing
│
├── frontend/
│   └── index.html              — Full mobile-style frontend (4 pages)
│
├── notebooks/
│   ├── project.ipynb           — Full pipeline: data generation, EDA, preprocessing, modelling
│   ├── Testcases.ipynb         — Model testing notebook
│   └── edge_case_tests_v2.py   — Automated edge case testing script (27 tests, 90% pass rate)
│
├── data/
│   └── cycle_data_full.csv     — Synthetic dataset (7200 rows × 34 columns)
│
└── README.md
```

---

## Dataset

The synthetic dataset was generated to reflect diverse real-world patterns:

| Property | Value |
|----------|-------|
| Total rows | 7,200 |
| Total columns | 34 |
| Users simulated | 200 |
| Cycles per user | 36 months |
| Age range | 13 to 45 |
| Conditions included | None, PCOS, Endometriosis, Thyroid |
| Spotting rate | ~55% of cycles |

Features are grouped into three categories:
- **Static features** — age, BMI, medical condition, contraceptive use
- **Dynamic features** — stress, sleep, mood, hydration, activity, cycle length
- **Lag and engineered features** — previous cycle history, streak length, rolling averages

---

## Model Performance

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| Random Forest Classifier | Spotting Yes/No | Accuracy | 65% |
| Random Forest Classifier | Spotting Yes/No | Recall | 74% |
| Random Forest Regressor | Gap in days | MAE | 0.577 |
| Random Forest Regressor | Gap in days | R² | 0.609 |

**Key finding from feature importance:**
- `personal_base_gap` is the strongest predictor of gap length (importance: 0.629)
- `personal_spotting_rate` is the strongest predictor of spotting occurrence (importance: 0.072)
- History features dominate over current cycle features

---

## Edge Case Testing

Automated edge case testing was performed across 5 groups covering 30 test cases:

| Group | Description |
|-------|-------------|
| Group 1 | Extreme personal profiles (age, BMI, condition) |
| Group 2 | Extreme stress and lifestyle inputs |
| Group 3 | History extremes (streak length, spotting rate) |
| Group 4 | Gap prediction extremes (base gap, variability) |
| Group 5 | Contradictory inputs (opposing signals) |

**Overall pass rate: 90% (27 of 30 tests)**

---

## How to Run Locally

**Prerequisites:**
- Python 3.9 or above
- Jupyter Notebook

**Step 1 — Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/cycleaware.git
cd cycleaware
```

**Step 2 — Install dependencies**
```bash
pip install -r backend/requirements.txt
```

**Step 3 — Start the backend**
```bash
cd backend
uvicorn app:app --reload
```

**Step 4 — Open the frontend**

Open `frontend/index.html` directly in your browser by double clicking it.

**Step 5 — Use the app**

Fill in the 3 sections and click Get Prediction. Keep the terminal open while using the app.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check if the server is running |
| `/predict` | POST | Submit cycle data and get prediction |

**Sample response from `/predict`:**
```json
{
  "spotting_probability": 0.653,
  "spotting_predicted": 1,
  "gap_days_predicted": 3,
  "safe_window_days": 2,
  "recommendation": "Spotting likely (65% probability). Bleeding expected in 3 days. Carry a pad within 2 days of spotting."
}
```

---

## Disclaimer

CycleAware is an indicative tool built for educational and personal use. It is not a medical device and should not be used as a substitute for professional medical advice. Predictions are based on patterns in synthetic data and may not reflect individual medical conditions accurately. Always consult a qualified healthcare provider for medical concerns.

---

## Acknowledgements

Built as an end-to-end machine learning project — from problem identification and synthetic data generation through exploratory data analysis, model training, API development, and frontend deployment.

---

*S. Lakshmi Tejaswi*
