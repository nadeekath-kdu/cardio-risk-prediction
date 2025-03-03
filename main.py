import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load the model and scaler with error handling
try:
    model = joblib.load("cardiovascular_risk_model.pkl")
    #scaler = joblib.load("scaler.pkl")  # Load the scaler if required
except Exception as e:
    print("Error loading model or scaler:", e)
    model = None
    scaler = None

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define input data format
class RiskInput(BaseModel):
    age: int
    sex: int
    cholesterol: float
    sbp: float
    diabetes: int
    smoking: int

# Risk levels mapping
risk_levels = ["<10%", "10% to <20%", "20% to <30%", "30% to <40%", ">=40%"]

# Function to map probability to risk level
def map_risk_level(probability_cvd):
    if probability_cvd < 0.10:
        return risk_levels[0]  # <10%
    elif 0.10 <= probability_cvd < 0.20:
        return risk_levels[1]  # 10% to <20%
    elif 0.20 <= probability_cvd < 0.30:
        return risk_levels[2]  # 20% to <30%
    elif 0.30 <= probability_cvd < 0.40:
        return risk_levels[3]  # 30% to <40%
    else:
        return risk_levels[4]  # >=40%

# Prediction endpoint
@app.post("/predict")
def predict_risk(data: RiskInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")

    try:

        # Convert input to DataFrame
        input_data = pd.DataFrame([data.dict()])

<<<<<<< HEAD
        # Ensure the columns are in the correct order
        expected_order = ['age', 'sex', 'cholesterol', 'sbp', 'diabetes', 'smoking']
        input_data = input_data[expected_order]

        # Scale the input data if required
        scaled_input = scaler.transform(input_data)
=======
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
>>>>>>> 712a71b5e67031656c76429fb7bfd62aba07657e

        print("Processed Data:")
        print(input_data)

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]

        # Log the prediction
        print("Prediction:", prediction)
        print("Prediction Probabilities:", prediction_proba)

        # Get the probability of CVD (class 1)
        probability_cvd = float(prediction_proba[1])

<<<<<<< HEAD
        # Map probability to risk level
        risk_level = map_risk_level(probability_cvd)

        # Return the results
        return {
            "prediction": int(prediction),  # 0 or 1
            "probability_no_cvd": float(prediction_proba[0]),  # Probability of no CVD
            "probability_cvd": probability_cvd,  # Probability of CVD
            "risk_level": risk_level  # Mapped risk level
        }
=======
        print("Prediction Level:", risk_level)

        return {"risk_level": risk_level}
>>>>>>> 712a71b5e67031656c76429fb7bfd62aba07657e
    except Exception as e:
        # Log the full error traceback
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")