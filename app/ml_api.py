# ===============================================================
# app/ml_api.py — FastAPI Backend for Renewal Prediction
# ===============================================================

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import os

app = FastAPI(title="Policy Renewal Prediction API")

# ===============================================================
# Load model and preprocessor
# ===============================================================
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
model_path = os.path.join(base_path, "models", "lgb_renewal_model.joblib")
preproc_path = os.path.join(base_path, "models", "preprocessing_objects.joblib")

model = joblib.load(model_path)
preprocessor = joblib.load(preproc_path)

# ===============================================================
# Root endpoint
# ===============================================================
@app.get("/")
def read_root():
    return {"message": "Welcome to the Renewal Prediction API 🚀"}

# ===============================================================
# Define request schema for JSON input
# ===============================================================
class InputData(BaseModel):
    data: List[dict]

# ===============================================================
# Predict endpoint (accepts JSON from Streamlit)
# ===============================================================
@app.post("/predict")
def predict(input_data: InputData):
    # Convert incoming data to DataFrame
    df = pd.DataFrame(input_data.data)

    # Preprocess
    if isinstance(preprocessor, dict):
        scaler = preprocessor["scaler"]
        features = preprocessor["features"]
        df = df[features]
        X_processed = scaler.transform(df)
    else:
        X_processed = preprocessor.transform(df)

    # Predict
    preds = model.predict_proba(X_processed)[:, 1]  # renewal probability

    # Return predictions as JSON
    return {"predictions": preds.tolist()}
