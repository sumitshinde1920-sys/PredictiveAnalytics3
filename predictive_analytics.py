# Predictive Analytics Using Historical Data
# VS Code Python Project
# ----------------------------------------
# Install Required Libraries:
# pip install pandas numpy matplotlib scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------------------
# STEP 1: Load Dataset
# ----------------------------------------

# Create sample historical data
# Example: Monthly Sales Data

data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Sales': [100, 120, 130, 150, 170, 180, 200, 210, 230, 250, 270, 300]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# ----------------------------------------
# STEP 2: Data Preprocessing
# ----------------------------------------

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Features and Target
X = df[['Month']]
y = df['Sales']

# ----------------------------------------
# STEP 3: Split Dataset
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------
# STEP 4: Train Predictive Model
# ----------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------------------
# STEP 5: Predictions
# ----------------------------------------

y_pred = model.predict(X_test)

# Predict future sales
future_months = pd.DataFrame({
    'Month': [13, 14, 15, 16]
})

future_predictions = model.predict(future_months)

# ----------------------------------------
# STEP 6: Model Evaluation
# ----------------------------------------

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Accuracy:")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# ----------------------------------------
# STEP 7: Visualization
# ----------------------------------------

plt.figure(figsize=(10, 5))

# Original Data
plt.scatter(df['Month'], df['Sales'], label='Actual Data')

# Regression Line
plt.plot(df['Month'], model.predict(X), label='Regression Line')

# Future Predictions
plt.scatter(
    future_months['Month'],
    future_predictions,
    label='Future Predictions'
)

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Predictive Analytics Using Historical Data")
plt.legend()

plt.show()

# ----------------------------------------
# STEP 8: Print Future Predictions
# ----------------------------------------

print("\nFuture Sales Predictions:")
for month, prediction in zip(future_months['Month'], future_predictions):
    print(f"Month {month}: {prediction:.2f}")