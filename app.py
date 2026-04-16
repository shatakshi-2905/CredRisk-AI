from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
import shap
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, OPTIONS
    allow_headers=["*"],
)
@app.get("/")
def home():
    return FileResponse("index.html")

# -----------------------------
# Load Artifacts
# -----------------------------
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

explainer = shap.TreeExplainer(model)

print("Model and encoders loaded")


# -----------------------------
# Feature Engineering
# -----------------------------
def engineer_features(data):

    data["loan_to_income"] = data["loan_amnt"] / data["person_income"]

    data["income_per_credit_year"] = (
        data["person_income"] /
        (data["cb_person_cred_hist_length"] + 1)
    )

    data["interest_income_ratio"] = (
        data["loan_int_rate"] /
        data["person_income"]
    )

    data["loan_income_interaction"] = (
        data["loan_amnt"] *
        data["loan_percent_income"]
    )

    return data


# -----------------------------
# Encoding
# -----------------------------
def encode_data(data):

    for col, encoder in encoders.items():

        if col in data:

            value = str(data[col])

            if value in encoder.classes_:
                data[col] = encoder.transform([value])[0]
            else:
                data[col] = 0

    return data


# -----------------------------
# Loan Grade Logic
# -----------------------------
def assign_grade(risk):

    if risk < 10:
        return "A"
    elif risk < 20:
        return "B"
    elif risk < 35:
        return "C"
    elif risk < 50:
        return "D"
    elif risk < 70:
        return "E"
    else:
        return "F"


# -----------------------------
# Risk Band
# -----------------------------
def risk_band(risk):

    if risk < 10:
        return "Very Low"
    elif risk < 25:
        return "Low"
    elif risk < 45:
        return "Medium"
    elif risk < 65:
        return "High"
    else:
        return "Very High"


# -----------------------------
# Credit Score
# -----------------------------
def credit_score(risk):

    score = 850 - (risk * 5)

    return int(max(300, min(score, 850)))


# -----------------------------
# Prediction API
# -----------------------------
@app.post("/predict")
def predict(input_data: dict):

    row = input_data.copy()

    # Feature engineering
    row = engineer_features(row)

    # Encode categoricals
    row = encode_data(row)

    # Convert to dataframe
    df = pd.DataFrame([row])

    # Ensure column order
    df = df[feature_columns]

    # Predict probability
    prob = float(model.predict_proba(df)[0][1])

    risk_percent = prob * 100

    # Loan grade
    grade = assign_grade(risk_percent)

    # Risk band
    band = risk_band(risk_percent)

    # Credit score
    score = credit_score(risk_percent)

    # -----------------------------
    # SHAP Explanation
    # -----------------------------
    shap_values = explainer.shap_values(df)

    impacts = []

    for i, feature in enumerate(feature_columns):

        impacts.append({
            "feature": feature,
            "impact": float(shap_values[0][i])
        })

    impacts = sorted(
        impacts,
        key=lambda x: abs(x["impact"]),
        reverse=True
    )[:8]

    # -----------------------------
    # Response
    # -----------------------------
    return {

        "risk_probability": round(prob, 4),

        "risk_percent": round(risk_percent, 2),

        "loan_grade": grade,

        "risk_band": band,

        "credit_score": score,

        "top_features": impacts
    }
