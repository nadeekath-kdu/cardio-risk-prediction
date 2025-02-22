import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load the model with error handling
try:
    model = joblib.load("cardiovascular_risk_model.pkl")
except Exception as e:
    print("Error loading model:", e)
    model = None  # Set to None if loading fails

# Initialize FastAPI app
app = FastAPI()

# Define input data format
class RiskInput(BaseModel):
    age: int
    cholesterol: float
    blood_pressure: float
    smoking: int
    diabetes: int

# Prediction endpoint
@app.post("/predict")
def predict_risk(data: RiskInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Make prediction
        prediction = model.predict(input_data)

        return {"risk_prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
