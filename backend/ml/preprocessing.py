# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# CONFIG - change if needed
INPUT_PATH = "./data/transactions.csv"
OUTPUT_PATH = "./data/transactions_processed.csv"
print("[INFO] Starting preprocessing...")

# 1) Load dataset
df = pd.read_csv(INPUT_PATH)
print(f"[INFO] Loaded data from {INPUT_PATH} — shape: {df.shape}")

# 2) Standardize column names
df.columns = (
    df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
)
print("[INFO] Columns standardized")

# 3) Handle missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
for col in categorical_cols:
    try:
        mode_val = df[col].mode(dropna=True)[0]
    except IndexError:
        mode_val = "missing"
    df[col] = df[col].fillna(mode_val)

print("[INFO] Missing values handled")

# 4) Remove duplicates
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
print(f"[INFO] Removed {before - after} duplicate rows")

# 5) Feature engineering
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df[["hour", "dayofweek", "month", "year"]] = (
        df[["hour", "dayofweek", "month", "year"]].fillna(-1).astype(int)
    )

if "transaction_amount" in df.columns:
    amt = pd.to_numeric(df["transaction_amount"], errors="coerce")
    unique_vals = amt.dropna().unique()
    if len(unique_vals) >= 4:
        df["amount_bin"] = pd.qcut(amt, q=4, labels=False, duplicates="drop")
    else:
        df["amount_bin"] = pd.cut(
            amt, bins=max(1, min(4, len(unique_vals))), labels=False
        )
    df["amount_bin"] = df["amount_bin"].fillna(-1).astype(int)

if "transaction_amount" in df.columns and "account_balance" in df.columns:
    amt = pd.to_numeric(df["transaction_amount"], errors="coerce").fillna(0.0)
    bal = pd.to_numeric(df["account_balance"], errors="coerce").fillna(0.0)
    df["amount_balance_ratio"] = amt / (bal + 1.0)

print("[INFO] Feature engineering done")

# 6) Encode categorical variables
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
le = LabelEncoder()
for col in categorical_cols:
    df[col] = df[col].fillna("missing").astype(str)
    try:
        df[col] = le.fit_transform(df[col])
    except Exception:
        mapping = {v: i for i, v in enumerate(df[col].astype(str).unique())}
        df[col] = df[col].astype(str).map(mapping)

print("[INFO] Categorical encoding completed")

# 7) Normalize numeric features (exclude target 'fraud_label' if present)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
scale_cols = [c for c in numeric_cols if c != "fraud_label"]
if scale_cols:
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    print("[INFO] Normalized numeric features (excluded 'fraud_label' if present)")
else:
    print("[INFO] No numeric features found to scale")

# 8) Final check & save
if "fraud_label" not in df.columns:
    raise ValueError("fraud_label column not found in dataset!")

df.to_csv(OUTPUT_PATH, index=False)
print(f"[INFO] Processed dataset saved at: {OUTPUT_PATH}")
print("[INFO] Preprocessing finished.")
