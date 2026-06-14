"""
Model extraction and training script from the mcd notebook
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
import os

# Load data
df = pd.read_csv(r"C:\Users\ALG\Downloads\archive\India_Menu.csv")

# Fill missing values
for col in df.select_dtypes(include=['float64','int64']).columns:
    df[col] = df[col].fillna(df[col].median())

# Get numeric columns
num = df.select_dtypes(include='number')

# Outlier detection and feature engineering
iqr_col = ['Sodium (mg)', 'Cholesterols (mg)']
for col in iqr_col:
    q1 = num[col].quantile(0.25)
    q3 = num[col].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    num[col] = num[col].clip(upper=upper)

# Log transformation
log_col = ['Total fat (g)', 'Total Sugars (g)', 'Protein (g)']
for col in log_col:
    num[col] = np.log1p(num[col])

# Add back to dataframe
df_processed = df.copy()
for col in num.columns:
    df_processed[col] = num[col]

# Features and target
# Input features for predicting Energy (kCal)
feature_cols = [
    'Protein (g)', 'Total fat (g)', 'Sat Fat (g)',
    'Trans fat (g)', 'Cholesterols (mg)', 'Total carbohydrate (g)',
    'Total Sugars (g)', 'Added Sugars (g)', 'Sodium (mg)'
]

# Target variable: Energy (kCal) to predict
X = df_processed[feature_cols]
y = df_processed['Energy (kCal)']

# Train-test split
x_new_train, x_new_test, y_new_train, y_new_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBRegressor for regression (predicting Energy/Calories)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(x_new_train, y_new_train)

# Print model performance
score = model.score(x_new_test, y_new_test)
print(f"Model R² Score: {score:.4f}")

# Save model
joblib.dump(model, 'mcd_model.pkl')
print("✅ Model saved as mcd_model.pkl")

# Verify
loaded_model = joblib.load('mcd_model.pkl')
print("✅ Model loaded and verified successfully")
