# ===============================================================
# app/ml_api.py — FastAPI Backend for Renewal Prediction + GenAI Chat
# ===============================================================

from dotenv import load_dotenv
from openai import OpenAI
import os

# Load environment variables and initialize OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib

# ===============================================================
# Initialize FastAPI app
# ===============================================================
app = FastAPI(title="Policy Renewal Prediction API")

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # can restrict to your Streamlit/Render URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    df = pd.DataFrame(input_data.data)

    if isinstance(preprocessor, dict):
        scaler = preprocessor["scaler"]
        features = preprocessor["features"]
        df = df[features]
        X_processed = scaler.transform(df)
    else:
        X_processed = preprocessor.transform(df)

    preds = model.predict_proba(X_processed)[:, 1]  # renewal probability
    return {"predictions": preds.tolist()}

# ===============================================================
# GenAI Chat endpoint
# ===============================================================
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant that explains customer "
                        "policy renewal predictions clearly and simply."
                    )
                },
                {"role": "user", "content": user_input}
            ]
        )

        reply = response.choices[0].message.content
        return JSONResponse(content={"reply": reply})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
