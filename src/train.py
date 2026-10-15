"""
train.py - Model Training Module

This module orchestrates the complete ML training pipeline:
1. Load and validate the dataset
2. Clean and preprocess the data
3. Split into train/test sets
4. Train Logistic Regression, Random Forest, and XGBoost
5. Evaluate all models
6. Select the best model
7. Save everything for the Streamlit app

Usage:
    python -m src.train
    python -m src.train --data data/churnData.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data_loader import load_dataset, validate_dataset, get_dataset_stats
from src.preprocessing import (
    clean_data, get_feature_columns, create_preprocessing_pipeline,
    prepare_features_target, save_preprocessing_artifacts,
    get_feature_names_after_transform
)
from src.evaluate import compare_models, select_best_model


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42       # Fixed seed for reproducibility
TEST_SIZE = 0.2         # 80% train, 20% test
MODELS_DIR = 'models'   # Directory to save trained models


def create_models():
    """
    Create the three classification models with sensible hyperparameters.
    
    Returns:
        dict: {model_name: model_instance}
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver='lbfgs',
            class_weight='balanced'  # Handle class imbalance
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1  # Use all CPU cores
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=1  # Will be set dynamically below
        )
    }
    return models


def run_training_pipeline(data_path='data/churnData.csv'):
    """
    Run the complete training pipeline from dataset to saved models.
    
    Parameters:
        data_path (str): Path to the CSV dataset.
        
    Returns:
        dict: Training results including metrics and best model info.
    """
    print("=" * 60)
    print("  Customer Churn Prediction - Training Pipeline")
    print("=" * 60)
    
    # ----------------------------------------------------------
    # Step 1: Load and Validate Dataset
    # ----------------------------------------------------------
    print("\n[1/8] Loading dataset...")
    df = load_dataset(data_path)
    print(f"  Loaded {len(df)} rows and {len(df.columns)} columns.")
    
    validation = validate_dataset(df)
    if not validation['is_valid']:
        print("  WARNING: Dataset validation issues:")
        for issue in validation['issues']:
            print(f"    - {issue}")
    else:
        print("  Dataset validation passed.")
    
    # ----------------------------------------------------------
    # Step 2: Dataset Statistics
    # ----------------------------------------------------------
    print("\n[2/8] Calculating dataset statistics...")
    target_col = validation.get('target_column')
    if target_col is None:
        # Try finding it after column normalization
        for col in df.columns:
            if 'churn' in col.lower():
                target_col = col
                break
    
    stats = get_dataset_stats(df, target_col if target_col else 'Churn_')
    print(f"  Rows: {stats['n_rows']}")
    print(f"  Features: {stats['n_features']}")
    print(f"  Missing values: {stats['total_missing']}")
    print(f"  Duplicates: {stats['n_duplicates']}")
    if stats.get('churn_rate') is not None:
        print(f"  Churn rate: {stats['churn_rate']:.1f}%")
    
    # ----------------------------------------------------------
    # Step 3: Clean Data
    # ----------------------------------------------------------
    print("\n[3/8] Cleaning data (removing duplicate samples for model training)...")
    df_clean = clean_data(df, drop_duplicates=True)
    print(f"  Cleaned dataset: {len(df_clean)} rows, {len(df_clean.columns)} columns.")
    
    # ----------------------------------------------------------
    # Step 4: Prepare Features and Target
    # ----------------------------------------------------------
    print("\n[4/8] Preparing features and target...")
    X, y, feature_names = prepare_features_target(df_clean)
    
    print(f"  Features shape: {X.shape}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    
    # Identify column types
    numerical_cols, categorical_cols = get_feature_columns(df_clean)
    print(f"  Numerical features ({len(numerical_cols)}): {numerical_cols}")
    print(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    # ----------------------------------------------------------
    # Step 5: Train/Test Split (BEFORE preprocessing to avoid leakage)
    # ----------------------------------------------------------
    print("\n[5/8] Splitting data (80/20, random_state={})...".format(RANDOM_STATE))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # ----------------------------------------------------------
    # Step 6: Preprocessing Pipeline (fit on train, transform both)
    # ----------------------------------------------------------
    print("\n[6/8] Building and fitting preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline(numerical_cols, categorical_cols)
    
    # Fit on training data only (avoid data leakage)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after transformation
    feature_names_transformed = get_feature_names_after_transform(
        preprocessor, numerical_cols, categorical_cols
    )
    
    print(f"  Processed features shape: {X_train_processed.shape}")
    
    # Save preprocessing artifacts
    save_preprocessing_artifacts(preprocessor, numerical_cols, categorical_cols,
                                 save_dir=MODELS_DIR)
    
    # ----------------------------------------------------------
    # Step 7: Train Models
    # ----------------------------------------------------------
    print("\n[7/8] Training models...")
    models = create_models()
    
    # Adjust XGBoost scale_pos_weight for class imbalance
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    if n_positive > 0:
        scale_pos_weight = n_negative / n_positive
        models['XGBoost'].set_params(scale_pos_weight=scale_pos_weight)
        print(f"  XGBoost scale_pos_weight set to {scale_pos_weight:.2f}")
    
    trained_models = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_processed, y_train)
        trained_models[name] = model
        print(f"    {name} trained successfully.")
    
    # ----------------------------------------------------------
    # Step 8: Evaluate and Compare Models
    # ----------------------------------------------------------
    print("\n[8/8] Evaluating models...")
    comparison_df, all_metrics = compare_models(
        trained_models, X_test_processed, y_test
    )
    
    print("\n  Model Comparison:")
    print(comparison_df.to_string())
    
    # Select best model
    best_name, best_model, selection_reason = select_best_model(
        comparison_df, trained_models
    )
    print(f"\n  Best Model: {best_name}")
    print(f"  Reason: {selection_reason}")
    
    # ----------------------------------------------------------
    # Save Everything
    # ----------------------------------------------------------
    print("\n  Saving trained models and results...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save all trained models
    joblib.dump(trained_models, os.path.join(MODELS_DIR, 'all_models.pkl'))
    
    # Save best model separately
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
    
    # Save test data for SHAP explanations
    joblib.dump({
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_test': y_test,
        'feature_names': feature_names_transformed,
        'X_train_raw': X_train,
        'X_test_raw': X_test,
    }, os.path.join(MODELS_DIR, 'test_data.pkl'))
    
    # Save evaluation results as JSON
    eval_results = {
        'comparison': comparison_df.reset_index().to_dict(orient='records'),
        'best_model_name': best_name,
        'selection_reason': selection_reason,
        'train_size': int(X_train.shape[0]),
        'test_size': int(X_test.shape[0]),
        'n_features': int(X_train_processed.shape[1]),
        'random_state': RANDOM_STATE,
        'test_ratio': TEST_SIZE,
    }
    
    # Convert numpy types for JSON serialization
    for record in eval_results['comparison']:
        for key, val in record.items():
            if isinstance(val, (np.floating, np.integer)):
                record[key] = float(val)
    
    with open(os.path.join(MODELS_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\n  All files saved to '{MODELS_DIR}/' directory:")
    for fname in os.listdir(MODELS_DIR):
        fpath = os.path.join(MODELS_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    - {fname} ({size_kb:.1f} KB)")
    
    print("\n" + "=" * 60)
    print("  Training complete! You can now run the Streamlit app:")
    print("  streamlit run app.py")
    print("=" * 60)
    
    return eval_results


# ============================================================
# Command-line entry point
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train churn prediction models.'
    )
    parser.add_argument(
        '--data', type=str, default='data/churnData.csv',
        help='Path to the dataset CSV file (default: data/churnData.csv)'
    )
    args = parser.parse_args()
    
    run_training_pipeline(data_path=args.data)
