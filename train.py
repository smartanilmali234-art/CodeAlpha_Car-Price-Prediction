import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("car_data.csv")  # change file name if needed

# ----------------------------
# 2. Basic preprocessing
# ----------------------------
# Example columns (adjust to your dataset):
# 'Year', 'Kms_Driven', 'Fuel_Type', 'Price'

df = df.dropna()

# Convert categorical columns if needed
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Price", axis=1)
y = df["Price"]

# ----------------------------
# 3. Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 4. Train model
# ----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# 5. Evaluate model
# ----------------------------
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

print("Model Accuracy (R2 Score):", score)

# ----------------------------
# 6. Save model safely
# ----------------------------
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)  # FIX: creates folder if not exists

model_path = os.path.join(model_dir, "model.pkl")
joblib.dump(model, model_path)

print(f"Model saved successfully at: {model_path}")columns.pkl")

print("✅ Model trained and saved successfully!")
