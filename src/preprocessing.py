"""
preprocessing.py - Feature Preprocessing Module

This module handles data cleaning, feature engineering, and building
the sklearn preprocessing pipeline for the churn prediction project.

Actual Dataset Columns (Pakistan Telecom):
- Gender: Male/Female
- Churn: Yes/No (TARGET)
- Network: Jazz/Warid, Zong, Ufone, Telenor
- relation_durattion: Duration categories
- service_type: Pre Paid / Post Paid
- source_of_attraction: How customer was attracted
- prferred_communication: messages, Audio calling, Video Calling
- satisfy_with_call_rate: Yes/No
- contact_with_cust_care: Yes/No
- rate_of_cust_care_calling: Frequency categories
- reascon_of_contact_cc: Reason for contacting customer care
- monthly_usage_rs: Numerical (50, 300, 500, 800)
- Happy_with_intrnet_quality: Yes/No
- 4G_available: Yes/No
- satisfy_with_voice_quality: Yes/No
- satisfy_with_signal_strength: Yes/No
- pleaseed_with_prices: Yes/No
- satisfy_with_billing_process: Yes/No
- recommend_to_others: Yes/No
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os


# ============================================================
# Column Definitions (from actual Pakistan Telecom dataset)
# ============================================================

# Target column
TARGET_COLUMN = 'Churn'

# Binary Yes/No columns (will be encoded to 1/0)
BINARY_COLUMNS = [
    'Gender',
    'satisfy_with_call_rate',
    'contact_with_cust_care',
    'Happy_with_intrnet_quality',
    '4G_available',
    'satisfy_with_voice_quality',
    'satisfy_with_signal_strength',
    'pleaseed_with_prices',
    'satisfy_with_billing_process',
    'recommend_to_others'
]

# Multi-class categorical columns (will be ordinal encoded)
MULTI_CATEGORICAL = [
    'Network',
    'relation_durattion',
    'service_type',
    'source_of_attraction',
    'prferred_communication',
    'rate_of_cust_care_calling',
    'reascon_of_contact_cc'
]

# Numerical columns
NUMERICAL_COLUMNS = ['monthly_usage_rs']


def clean_data(df, drop_duplicates=False):
    """
    Clean the dataset:
    - Strip whitespace
    - Optionally remove duplicate rows (for training pipeline)
    - Encode binary columns (Yes/No → 1/0, Male/Female → 1/0)
    - Encode target column if present
    
    Parameters:
        df (pd.DataFrame): Raw dataset.
        drop_duplicates (bool): Whether to remove duplicate rows (default: False for inference).
        
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    df = df.copy()
    
    # Strip whitespace from all string/object columns
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()
    
    # Remove duplicate rows only when explicitly requested (e.g. during model training)
    if drop_duplicates:
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        if n_removed > 0:
            print(f"  Removed {n_removed} duplicate rows. ({n_before} → {len(df)})")
    
    # Encode target column (Yes → 1, No → 0)
    if TARGET_COLUMN in df.columns:
        if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
            mapped_t = df[TARGET_COLUMN].astype(str).str.strip().str.lower().map({
                'yes': 1, 'no': 0, '1': 1, '0': 0, 'true': 1, 'false': 0
            })
            df[TARGET_COLUMN] = mapped_t.fillna(0).astype(int)
    
    # Encode Gender (Male → 1, Female → 0)
    if 'Gender' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['Gender']):
            mapped_g = df['Gender'].astype(str).str.strip().str.lower().map({
                'male': 1, 'female': 0, '1': 1, '0': 0
            })
            df['Gender'] = mapped_g.fillna(0).astype(int)
    
    # Encode binary Yes/No columns
    binary_yes_no = [col for col in BINARY_COLUMNS if col != 'Gender' and col in df.columns]
    for col in binary_yes_no:
        if not pd.api.types.is_numeric_dtype(df[col]):
            mapped_b = df[col].astype(str).str.strip().str.lower().map({
                'yes': 1, 'no': 0, '1': 1, '0': 0, 'true': 1, 'false': 0
            })
            df[col] = mapped_b.fillna(0).astype(int)
    
    return df


def get_feature_columns(df):
    """
    Identify numerical and categorical feature columns in the cleaned dataset.
    Adapts to whatever columns are actually present.
    
    Parameters:
        df (pd.DataFrame): Cleaned dataset.
        
    Returns:
        tuple: (numerical_cols, categorical_cols) lists.
    """
    # Exclude target column
    exclude = {TARGET_COLUMN}
    feature_cols = [c for c in df.columns if c not in exclude]
    
    numerical_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return numerical_cols, categorical_cols


def create_preprocessing_pipeline(numerical_cols, categorical_cols):
    """
    Create a sklearn ColumnTransformer preprocessing pipeline.
    
    - Numerical features: Impute (median) + StandardScaler
    - Categorical features: Impute (most frequent) + OrdinalEncoder
    
    Parameters:
        numerical_cols (list): List of numerical column names.
        categorical_cols (list): List of categorical column names.
        
    Returns:
        ColumnTransformer: The preprocessing pipeline.
    """
    transformers = []
    
    # Numerical pipeline
    if numerical_cols:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('numerical', num_pipeline, numerical_cols))
    
    # Categorical pipeline
    if categorical_cols:
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(
                handle_unknown='use_encoded_value', 
                unknown_value=-1
            ))
        ])
        transformers.append(('categorical', cat_pipeline, categorical_cols))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop any columns not explicitly handled
    )
    
    return preprocessor


def prepare_features_target(df):
    """
    Separate features (X) and target (y) from the cleaned dataset.
    
    Parameters:
        df (pd.DataFrame): Cleaned dataset.
        
    Returns:
        tuple: (X, y, feature_names)
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in the dataset. "
            f"Available columns: {list(df.columns)}"
        )
    
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])
    
    return X, y, list(X.columns)


def get_feature_names_after_transform(preprocessor, numerical_cols, categorical_cols):
    """
    Get feature names after preprocessing transformation.
    
    Parameters:
        preprocessor (ColumnTransformer): Fitted preprocessor.
        numerical_cols (list): Original numerical column names.
        categorical_cols (list): Original categorical column names.
        
    Returns:
        list: Feature names after transformation.
    """
    feature_names = list(numerical_cols) + list(categorical_cols)
    return feature_names


def save_preprocessing_artifacts(preprocessor, numerical_cols, categorical_cols,
                                  feature_names=None, save_dir='models'):
    """
    Save the preprocessing pipeline and column information.
    
    Parameters:
        preprocessor: Fitted ColumnTransformer.
        numerical_cols (list): Numerical column names.
        categorical_cols (list): Categorical column names.
        feature_names (list, optional): All feature names in order.
        save_dir (str): Directory to save artifacts.
    """
    os.makedirs(save_dir, exist_ok=True)
    if feature_names is None:
        feature_names = list(numerical_cols) + list(categorical_cols)
    
    artifacts = {
        'preprocessor': preprocessor,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'feature_names': feature_names
    }
    
    joblib.dump(artifacts, os.path.join(save_dir, 'preprocessing_pipeline.pkl'))
    print(f"  Preprocessing artifacts saved to '{save_dir}/preprocessing_pipeline.pkl'")


def load_preprocessing_artifacts(save_dir='models'):
    """
    Load saved preprocessing pipeline and column information.
    
    Parameters:
        save_dir (str): Directory containing saved artifacts.
        
    Returns:
        dict: Dictionary with preprocessor, column names, and feature names.
    """
    path = os.path.join(save_dir, 'preprocessing_pipeline.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Preprocessing pipeline not found at '{path}'. "
            f"Please train the models first by running: python -m src.train"
        )
    return joblib.load(path)


def find_target_in_df(df):
    """
    Find the target column in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        str: Target column name.
    """
    if TARGET_COLUMN in df.columns:
        return TARGET_COLUMN
    for col in df.columns:
        if 'churn' in col.lower():
            return col
    return 'Churn'

