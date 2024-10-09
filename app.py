import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

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
condition = st.selectbox('Condition', [0, 1])  # Assuming condition is an integer from 1 to 5

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

    # Predict the result
    try:
        model = models[model_name]
        prediction = model.predict(input_df)[0]

        # Load y_test for evaluation
        y_test = pd.read_csv('y_test.csv')
        X_test = pd.read_csv('X_test.csv')

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.success(f"Predicted house price: ${prediction:,.2f}")
        st.info(f"Model evaluation metrics:\n- MAE: {mae:.4f}\n- R²: {r2:.4f}\n- RMSE: {rmse:.4f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

