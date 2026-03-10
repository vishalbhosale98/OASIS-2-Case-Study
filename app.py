from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


# Load Model and Scaler
model = joblib.load("trained_models/logistic_model.pkl")
scaler = joblib.load("trained_models/minmax_scaler.pkl")


# Feature Order
feature_cols = [
    # 'Age',
    # 'EDUC',
    'SES',
    # 'MR_Delay',
    # 'eTIV',
    # 'nWBV',
    # 'ASF',
    'time_since_first_visit',
    'nWBV_change',
    'brain_atrophy_rate',
    'visit_count'
]


app = FastAPI(title="Alzheimer Risk Prediction API")


# Input Schema
class PatientData(BaseModel):

    Age: float
    EDUC: float
    SES: float
    MR_Delay: float
    eTIV: float
    nWBV: float
    ASF: float
    time_since_first_visit: float
    nWBV_change: float
    brain_atrophy_rate: float
    visit_count: float


@app.get("/")
def home():
    return {"message": "Alzheimer Prediction API is running"}


# Prediction Endpoint
@app.post("/predict")
def predict(data: PatientData):

    # Convert input to dataframe
    input_data = pd.DataFrame([data.dict()])
    
    # Ensure correct feature order
    input_data = input_data[feature_cols]

    # Apply MinMax Scaling
    input_scaled = scaler.transform(input_data)

    input_scaled = pd.DataFrame(
        input_scaled,
        columns=feature_cols
    )

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "probability_of_dementia": round(float(probability), 4)
    }