import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("data/car_data.csv")
print("✅ Dataset Loaded")

# -----------------------------
# Clean Column Names
# -----------------------------
data.columns = data.columns.str.strip().str.lower()
print("Columns:", data.columns.tolist())

# -----------------------------
# Feature Engineering
# -----------------------------
# Create car age
data['car_age'] = 2024 - data['year']

# -----------------------------
# Encode Categorical Data
# -----------------------------
# Fuel Type
data['fuel_type'] = data['fuel_type'].str.strip().str.lower()
data['fuel_type'] = data['fuel_type'].map({'petrol': 0, 'diesel': 1, 'cng': 2})

# Transmission
data['transmission'] = data['transmission'].str.strip().str.lower()
data['transmission'] = data['transmission'].map({'manual': 0, 'automatic': 1})

# Selling Type
data['selling_type'] = data['selling_type'].str.strip().str.lower()
data['selling_type'] = data['selling_type'].map({'dealer': 0, 'individual': 1})

# -----------------------------
# Drop unnecessary columns
# -----------------------------
data = data.drop(['car_name'], axis=1)

# -----------------------------
# Features & Target
# -----------------------------
X = data.drop('selling_price', axis=1)
y = data['selling_price']

# Fill missing values
X = X.fillna(0)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model Training
# -----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/model.pkl")
joblib.dump(X.columns.tolist(), "model/columns.pkl")

print("✅ Model trained and saved successfully!")