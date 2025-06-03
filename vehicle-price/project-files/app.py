import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("best_vehicle_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🚗 Vehicle Price Predictor")
st.markdown("Predict vehicle prices based on specifications")

# Load input column names
input_cols = joblib.load("input_columns.pkl")

# User input form
with st.form("input_form"):
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2024)
    mileage = st.number_input("Mileage (in miles)", min_value=0, value=10)
    cylinders = st.selectbox("Cylinders", [4, 6, 8])
    doors = st.selectbox("Number of Doors", [2, 4])
    
    # Simulated encoding inputs
    fuel = st.selectbox("Fuel Type", ['Gasoline', 'Diesel', 'Electric', 'Hybrid', 'Unknown'])
    transmission = st.selectbox("Transmission", ['Automatic', 'Manual', 'CVT', 'Unknown'])
    body = st.selectbox("Body Type", ['SUV', 'Sedan', 'Pickup Truck', 'Unknown'])
    drivetrain = st.selectbox("Drivetrain", ['Four-wheel Drive', 'Rear-wheel Drive', 'All-wheel Drive', 'Front-wheel Drive', 'Unknown'])

    submit = st.form_submit_button("Predict Price")

# Prepare input and predict
if submit:
    data = {
        'year': year,
        'mileage': mileage,
        'cylinders': cylinders,
        'doors': doors,
        'fuel_' + fuel: 1,
        'transmission_' + transmission: 1,
        'body_' + body: 1,
        'drivetrain_' + drivetrain: 1
    }

    # Initialize with 0s for all expected columns
    full_input = {col: 0 for col in input_cols}
    for key, val in data.items():
        if key in full_input:
            full_input[key] = val
    X_input = pd.DataFrame([full_input])
    
    # Scale and predict
    X_scaled = scaler.transform(X_input)
    prediction = model.predict(X_scaled)[0]
    st.success(f"💰 Predicted Price: ${prediction:,.2f}")
