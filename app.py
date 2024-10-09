import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Load the trained models
model_dir = os.path.join(os.path.dirname(__file__), 'models')
lr_model = joblib.load(os.path.join(model_dir, 'linear_regression_model.joblib'))
ridge_model = joblib.load(os.path.join(model_dir, 'ridge_regression_model.joblib'))
mlp_model = joblib.load(os.path.join(model_dir, 'mlp_regressor_model.joblib'))
stacking_model = joblib.load(os.path.join(model_dir, 'stacking_regressor_model.joblib'))

# Create a dictionary for the models
models = {
    'Linear Regression': lr_model,
    'Ridge Regression': ridge_model,
    'Neural Network': mlp_model,
    'Stacking Regressor': stacking_model
}

# Start Streamlit app
st.title("House Price Prediction App")

# Input form for user data
st.header("Input data for prediction")

# Create input fields for user
bedroom = st.number_input('Bedroom', value=0)
space = st.number_input('Space (sq ft)', value=0)
room = st.number_input('Room', value=0)
lot = st.number_input('Lot (sq ft)', value=0)
tax = st.number_input('Tax ($)', value=0)
bathroom = st.number_input('Bathroom', value=0)
garage = st.number_input('Garage (cars)', value=0)
condition = st.selectbox('Condition', [1, 2, 3, 4, 5])  # Assuming condition is an integer from 1 to 5

# Select model
model_name = st.selectbox(
    'Prediction Model',
    ['Linear Regression', 'Ridge Regression', 'Neural Network', 'Stacking Regressor']
)

# When the user clicks the "Predict" button
if st.button('Predict'):
    # Create DataFrame from user input
    input_data = {
        'Bedroom': bedroom,
        'Space': space,
        'Room': room,
        'Lot': lot,
        'Tax': tax,
        'Bathroom': bathroom,
        'Garage': garage,
        'Condition': condition
    }

    # Convert dictionary to DataFrame for prediction
    input_df = pd.DataFrame([input_data])

    # Dự đoán kết quả
    try:
        model = models[model_name]
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted house price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

    # Add model reliability information
    model_scores = {
        'Linear Regression': 69043.17,
        'Ridge Regression': 69043.17,
        'Neural Network': 56023.45,
        'Stacking Regressor': 55012.34
    }
    confidence = model_scores.get(model_name)
    st.info(f"Model confidence (RMSE): {confidence}")

