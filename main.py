import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model
model = joblib.load("cardiovascular_risk_model.pkl")

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
    # Convert input to DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    return {"risk_prediction": int(prediction[0])}
