import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Paths
DATA_PATH = "./data/transactions_processed.csv"
MODEL_PATH = "./models/fraud_model.pkl"

print("[INFO] Starting training...")

# Load processed data
df = pd.read_csv(DATA_PATH)
print(f"[INFO] Loaded processed data — shape: {df.shape}")

# Split features and target
X = df.drop(columns=["fraud_label"], errors="ignore")
y = df["fraud_label"]

# Keep only numeric features (drop datetime / strings)
X = X.select_dtypes(include=["number"])
print(f"[INFO] Training with {X.shape[1]} numeric features.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("[INFO] Model training complete.")

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"[INFO] Model saved at {MODEL_PATH}")
