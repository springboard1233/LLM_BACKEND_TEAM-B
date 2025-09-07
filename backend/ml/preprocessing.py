# improved_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# CONFIG
INPUT_PATH = "./ml/data/transactions.csv"
OUTPUT_PATH = "./ml/data/transactions_processed.csv"
MODELS_DIR = "./ml/models/"

def improved_preprocessing():
    print("[INFO] Starting improved preprocessing...")
    
    # 1) Load dataset
    df = pd.read_csv(INPUT_PATH)
    print(f"[INFO] Loaded data from {INPUT_PATH} — shape: {df.shape}")
    
    # 2) Standardize column names (your existing code is good)
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
    )
    print("[INFO] Columns standardized")
    print(f"[INFO] Available columns: {list(df.columns)}")
    
    # 3) Handle missing values - more conservative approach
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        # Use median for skewed financial data
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")  # More explicit than "missing"
    
    print("[INFO] Missing values handled")
    
    # 4) Remove duplicates (your code is good)
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"[INFO] Removed {before - after} duplicate rows")
    
    # 5) ENHANCED Feature Engineering - Fraud-specific features
    print("[INFO] Creating fraud-specific features...")
    
    # DateTime features (improved)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
        df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)
        
        # Drop original timestamp to avoid data leakage
        df = df.drop("timestamp", axis=1)
    
    # Transaction amount features (more sophisticated)
    if "transaction_amount" in df.columns:
        amt = pd.to_numeric(df["transaction_amount"], errors="coerce")
        
        # Amount categories (more meaningful)
        df["is_small_amount"] = (amt <= 10).astype(int)
        df["is_large_amount"] = (amt >= 500).astype(int)
        df["amount_log"] = np.log1p(amt)  # Log transform for skewed data
        
        # Amount vs balance ratio (your idea was good)
        if "account_balance" in df.columns:
            bal = pd.to_numeric(df["account_balance"], errors="coerce").fillna(0.0)
            df["amount_to_balance_ratio"] = amt / (bal + 1.0)
            df["is_high_ratio"] = (df["amount_to_balance_ratio"] > 0.1).astype(int)
    
    # Risk-based features
    if "risk_score" in df.columns:
        risk = pd.to_numeric(df["risk_score"], errors="coerce")
        df["is_high_risk"] = (risk > 0.7).astype(int)
        df["is_low_risk"] = (risk < 0.3).astype(int)
    
    # Transaction pattern features
    if "daily_transaction_count" in df.columns:
        daily_count = pd.to_numeric(df["daily_transaction_count"], errors="coerce")
        df["is_high_frequency"] = (daily_count > 10).astype(int)
    
    if "failed_transaction_count_7d" in df.columns:
        failed_count = pd.to_numeric(df["failed_transaction_count_7d"], errors="coerce")
        df["has_recent_failures"] = (failed_count > 0).astype(int)
    
    # Device and location risk
    if "ip_address_flag" in df.columns:
        df["suspicious_ip"] = pd.to_numeric(df["ip_address_flag"], errors="coerce").fillna(0)
    
    if "previous_fraudulent_activity" in df.columns:
        df["has_fraud_history"] = pd.to_numeric(df["previous_fraudulent_activity"], errors="coerce").fillna(0)
    
    print("[INFO] Enhanced feature engineering completed")
    
    # 6) Smart categorical encoding
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Save encoders for later use
    encoders = {}
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for col in categorical_cols:
        if col == "fraud_label":  # Skip target variable
            continue
            
        le = LabelEncoder()
        df[col] = df[col].fillna("Unknown").astype(str)
        df[col] = le.fit_transform(df[col])
        
        # Save encoder
        encoders[col] = le
        encoder_path = os.path.join(MODELS_DIR, f"{col}_encoder.pkl")
        joblib.dump(le, encoder_path)
    
    print(f"[INFO] Categorical encoding completed for {len(categorical_cols)} columns")
    
    # 7) Create interaction features (NEW - important for fraud detection)
    print("[INFO] Creating interaction features...")
    
    # High-risk combinations
    if all(col in df.columns for col in ["is_night", "is_large_amount"]):
        df["night_large_transaction"] = df["is_night"] * df["is_large_amount"]
    
    if all(col in df.columns for col in ["is_weekend", "has_recent_failures"]):
        df["weekend_with_failures"] = df["is_weekend"] * df["has_recent_failures"]
    
    # 8) Feature selection - remove low-variance features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if "fraud_label" in numeric_features:
        numeric_features.remove("fraud_label")
    
    # Remove features with very low variance (likely not useful)
    low_variance_features = []
    for col in numeric_features:
        if df[col].var() < 0.01:  # Very low variance threshold
            low_variance_features.append(col)
    
    if low_variance_features:
        print(f"[INFO] Removing {len(low_variance_features)} low-variance features")
        df = df.drop(columns=low_variance_features)
    
    # 9) Final validation
    if "fraud_label" not in df.columns:
        raise ValueError("fraud_label column not found in dataset!")
    
    # Check for any remaining issues
    print(f"[INFO] Final dataset shape: {df.shape}")
    print(f"[INFO] Features: {df.shape[1] - 1}")  # -1 for target variable
    print(f"[INFO] Fraud rate: {df['fraud_label'].mean():.3f}")
    
    # 10) Save processed data
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Processed dataset saved at: {OUTPUT_PATH}")
    
    # Save feature info for later reference
    feature_info = {
        'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_features_encoded': list(encoders.keys()),
        'total_features': df.shape[1] - 1,
        'fraud_rate': float(df['fraud_label'].mean())
    }
    
    feature_info_path = os.path.join(MODELS_DIR, "feature_info.pkl")
    joblib.dump(feature_info, feature_info_path)
    print(f"[INFO] Feature info saved at: {feature_info_path}")
    
    print("[SUCCESS] Improved preprocessing completed!")
    return df, encoders, feature_info

if __name__ == "__main__":
    try:
        df, encoders, feature_info = improved_preprocessing()
        print(f"\n[SUMMARY] Created {feature_info['total_features']} features for training")
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {str(e)}")
        import traceback
        traceback.print_exc()