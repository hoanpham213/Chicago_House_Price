import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.impute import SimpleImputer
import joblib
import os

# Load the data from the CSV file
df = pd.read_csv('chicago_house.csv')

# Prepare data for training
X = df[['Bedroom', 'Space', 'Room', 'Lot', 'Tax', 'Bathroom', 'Garage', 'Condition']]
y = df['Price']

# Check for NaN values in the features
print("Checking for NaN values in the features...")
print(X.isnull().sum())

# Handle missing values by using SimpleImputer
imputer = SimpleImputer(strategy='mean')  # You can use 'median' or 'most_frequent' if you prefer
X_imputed = imputer.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Define and train your models
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=2000, random_state=42)
mlp_model.fit(X_train, y_train)

estimators = [
    ('lr', lr_model),
    ('ridge', ridge_model),
    ('mlp', mlp_model)
]

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge()
)
stacking_model.fit(X_train, y_train)

# Save the models
if not os.path.exists('models'):
    os.makedirs('models')
joblib.dump(lr_model, 'models/linear_regression_model.joblib')
joblib.dump(ridge_model, 'models/ridge_regression_model.joblib')
joblib.dump(mlp_model, 'models/mlp_regressor_model.joblib')
joblib.dump(stacking_model, 'models/stacking_regressor_model.joblib')

print("Models trained and saved successfully!")
