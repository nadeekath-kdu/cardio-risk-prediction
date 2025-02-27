import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load the model with error handling
try:
    model = joblib.load("cardiovascular_risk_model.pkl")
except Exception as e:
    print("Error loading model:", e)
    model = None  # Set to None if loading fails

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can specify specific origins here)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define input data format
class RiskInput(BaseModel):
    age: int
    cholesterol: float
    blood_pressure: float
    smoking: int
    diabetes: int
    sex: int

# Prediction endpoint
@app.post("/predict")
def predict_risk(data: RiskInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        print("Raw Data:", data.dict())

        # Convert input to DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Rename features to match the trained model
        input_data = input_data.rename(columns={
            "age": "Age", 
            "sex": "Sex", 
            "cholesterol": "Cholesterol",
            "diabetes": "Diabetes",
            "smoking": "Smoking",
            "blood_pressure": "sbp"
        })
        expected_order =  ['Age', 'Sex' ,'Cholesterol', 'sbp' ,'Diabetes' ,'Smoking']
        input_data = input_data[expected_order] 

        print("Processed Data:")
        print(input_data)

        # Make prediction
        prediction_code = model.predict(input_data)[0]

        # Log the prediction
        print("Prediction Code:", prediction_code)

        # Map the prediction code to the risk level
        risk_levels = ["<10%", "10% to <20%", "20% to <30%", "30% to <40%", ">=40%"]
        risk_level = risk_levels[prediction_code]

        return {"risk_level": risk_level}
    except Exception as e:
        # Log the full error traceback
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

