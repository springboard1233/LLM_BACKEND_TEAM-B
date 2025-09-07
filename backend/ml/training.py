# improved_training.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Paths
DATA_PATH = "./ml/data/transactions_processed.csv"
MODELS_DIR = "./ml/models/fraud_models"

def train_multiple_models():
    print("[INFO] Starting enhanced training...")
    
    # Load processed data
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded data with shape: {df.shape}")
    
    # Prepare features and target
    X = df.drop(columns=["fraud_label"], errors="ignore")
    y = df["fraud_label"]
    X = X.select_dtypes(include=["number"])
    
    print(f"[INFO] Training with {X.shape[1]} features")
    print(f"[INFO] Class distribution - Non-fraud: {(y==0).sum()}, Fraud: {(y==1).sum()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle class imbalance with SMOTE
    print("[INFO] Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE - Samples: {X_train_balanced.shape[0]}")
    
    # Calculate class weights for models that support it
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    models = {}
    
    # 1. Enhanced Random Forest
    print("\n[INFO] Training Enhanced Random Forest...")
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    }
    
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='f1', n_jobs=-1)
    rf_grid.fit(X_train_balanced, y_train_balanced)
    
    models['enhanced_random_forest'] = rf_grid.best_estimator_
    print(f"[INFO] Best RF F1 Score: {rf_grid.best_score_:.4f}")
    
    # 2. Gradient Boosting
    print("\n[INFO] Training Gradient Boosting...")
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='f1', n_jobs=-1)
    gb_grid.fit(X_train_balanced, y_train_balanced)
    
    models['gradient_boosting'] = gb_grid.best_estimator_
    print(f"[INFO] Best GB F1 Score: {gb_grid.best_score_:.4f}")
    
    # 3. Logistic Regression with class weights
    print("\n[INFO] Training Logistic Regression...")
    lr_params = {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': ['balanced']
    }
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_grid = GridSearchCV(lr, lr_params, cv=3, scoring='f1', n_jobs=-1)
    lr_grid.fit(X_train, y_train)  # Using original data for LR
    
    models['logistic_regression'] = lr_grid.best_estimator_
    print(f"[INFO] Best LR F1 Score: {lr_grid.best_score_:.4f}")
    
    # Evaluate all models on test set
    print("\n" + "="*60)
    print("MODEL COMPARISON ON TEST SET")
    print("="*60)
    
    best_model = None
    best_f1 = 0
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        print(f"\n{name.upper()}:")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = (name, model)
    
    # Save all models
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for name, model in models.items():
        model_path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
        joblib.dump(model, model_path)
        print(f"[INFO] {name} saved to {model_path}")
    
    # Save the best model as the main fraud model
    main_model_path = os.path.join(MODELS_DIR, "fraud_model.pkl")
    joblib.dump(best_model[1], main_model_path)
    print(f"\n[SUCCESS] Best model ({best_model[0]}) saved as main fraud model")
    print(f"[SUCCESS] Best F1 Score: {best_f1:.4f}")
    
    return models, best_model

if __name__ == "__main__":
    try:
        models, best_model = train_multiple_models()
        print("\n[SUCCESS] Enhanced training completed!")
    except Exception as e:
        print(f"[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()