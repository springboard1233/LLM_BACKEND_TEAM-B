# evaluation.py
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
DATA_PATH = "./ml/data/transactions_processed.csv"
MODEL_PATH = "./ml/models/fraud_models/fraud_model.pkl"

def evaluate_model():
    print("[INFO] Starting model evaluation...")
    
    # Load processed data and model
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Loaded data with shape: {df.shape}")
    
    # Prepare features and target
    X = df.drop(columns=["fraud_label"], errors="ignore")
    y = df["fraud_label"]
    
    # Keep only numeric features
    X = X.select_dtypes(include=["number"])
    print(f"[INFO] Using {X.shape[1]} features for evaluation")
    
    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_proba)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"ROC-AUC:   {auc:.4f} ({auc*100:.2f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print("\nCONFUSION MATRIX:")
    print(f"True Negatives:  {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives:  {cm[1,1]}")
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    print(f"\nCROSS-VALIDATION F1-SCORES:")
    print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Individual scores: {cv_scores}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTOP 10 MOST IMPORTANT FEATURES:")
        print(feature_importance.head(10).to_string(index=False))
    
    # Detailed classification report
    print("\nDETAILED CLASSIFICATION REPORT:")
    print(classification_report(y, y_pred))
    
    # Class distribution
    print(f"\nCLASS DISTRIBUTION:")
    print(f"Non-fraud (0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    print(f"Fraud (1):     {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'feature_importance': feature_importance if hasattr(model, 'feature_importances_') else None
    }

if __name__ == "__main__":
    try:
        results = evaluate_model()
        print(f"\n[SUCCESS] Evaluation completed successfully!")
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {str(e)}")