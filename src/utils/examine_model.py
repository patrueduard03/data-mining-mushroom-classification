"""
Script to examine the model features and data structure
"""
import pandas as pd
from catboost import CatBoostClassifier
import os

# Load the data
df = pd.read_csv('./data/mushroom.csv')
print("Dataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())

# Check unique values for categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical columns:", categorical_columns)

# Display unique values for each categorical column
for col in categorical_columns:
    unique_vals = df[col].unique()
    print(f"\n{col}: {len(unique_vals)} unique values")
    print(f"Values: {unique_vals}")

# Check numeric columns
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"\nNumeric columns: {numeric_columns}")

# Load the model if it exists
model_path = './models/catboost_model.cbm'
if os.path.exists(model_path):
    model = CatBoostClassifier()
    model.load_model(model_path)
    print("\nModel loaded successfully!")
    feature_names = model.feature_names_
    print(f"Feature names: {feature_names}")
    if feature_names:
        print(f"Number of features: {len(feature_names)}")
else:
    print(f"\nModel not found at {model_path}")
