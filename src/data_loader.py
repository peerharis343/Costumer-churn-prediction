"""
data_loader.py - Dataset Loading and Validation Module

This module handles loading the Pakistan Telecom churn dataset,
validating its structure, and providing summary statistics.

Actual Dataset Structure (Pakistan Telecom Data Set from Kaggle):
- 9,760 rows, 19 columns
- 18 categorical columns + 1 numerical column (monthly_usage_rs)
- Target column: 'Churn' (Yes/No)
- Networks: Jazz/Warid, Zong, Ufone, Telenor
- No missing values
- High number of duplicate rows (survey-style data)
"""

import pandas as pd
import numpy as np
import os


def load_dataset(data_path="data/churnData.csv"):
    """
    Load the telecom churn dataset from a CSV file.
    
    Parameters:
        data_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded dataset.
        
    Raises:
        FileNotFoundError: If the dataset file is not found.
        ValueError: If the file is empty or has invalid format.
    """
    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. "
            f"Please download the Pakistan Telecom dataset from Kaggle: "
            f"https://www.kaggle.com/datasets/zeshanali/pakistan-telecom-data-set "
            f"and place 'churnData.csv' in the 'data/' folder."
        )
    
    # Load the CSV file
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f"Error reading the CSV file: {e}")
    
    # Check if dataset is empty
    if df.empty:
        raise ValueError("The dataset is empty. Please check the CSV file.")
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Strip whitespace from string values
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    return df


def validate_dataset(df):
    """
    Validate that the dataset has the expected structure for churn prediction.
    
    Parameters:
        df (pd.DataFrame): The dataset to validate.
        
    Returns:
        dict: Validation results with 'is_valid' flag and 'issues' list.
    """
    issues = []
    
    # Check for target column
    target_col = find_target_column(df)
    if target_col is None:
        issues.append("No 'Churn' target column found in the dataset.")
    
    # Check minimum number of rows
    if len(df) < 100:
        issues.append(f"Dataset has only {len(df)} rows. Need at least 100 for training.")
    
    # Check minimum number of columns
    if len(df.columns) < 3:
        issues.append(f"Dataset has only {len(df.columns)} columns. Need at least 3.")
    
    # Check for duplicate rows (informational, not a blocker)
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        issues.append(
            f"Found {n_duplicates} duplicate rows ({n_duplicates/len(df)*100:.1f}% of data). "
            f"These will be removed during cleaning."
        )
    
    # Validation passes even if there are duplicates (they'll be handled)
    is_valid = target_col is not None and len(df) >= 100 and len(df.columns) >= 3
    
    return {
        'is_valid': is_valid,
        'issues': issues,
        'target_column': target_col
    }


def get_dataset_stats(df, target_col='Churn'):
    """
    Calculate summary statistics for the dataset.
    
    Parameters:
        df (pd.DataFrame): The dataset.
        target_col (str): Name of the target column.
        
    Returns:
        dict: Dictionary containing dataset statistics.
    """
    stats = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'n_features': len(df.columns) - 1,  # Exclude target
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'total_missing': int(df.isnull().sum().sum()),
        'n_duplicates': int(df.duplicated().sum()),
    }
    
    # Identify column types
    stats['numerical_columns'] = list(df.select_dtypes(include=[np.number]).columns)
    stats['categorical_columns'] = list(df.select_dtypes(include=['object', 'category']).columns)
    
    # Target column statistics
    if target_col and target_col in df.columns:
        churn_counts = df[target_col].value_counts()
        stats['churn_distribution'] = churn_counts.to_dict()
        
        # Calculate churn rate
        try:
            if df[target_col].dtype in ['int64', 'float64']:
                stats['churn_rate'] = float(df[target_col].mean() * 100)
            else:
                # String values like 'Yes'/'No'
                yes_count = df[target_col].apply(
                    lambda x: str(x).strip().lower() in ['yes', 'true', '1']
                ).sum()
                stats['churn_rate'] = float(yes_count / len(df) * 100)
        except Exception:
            stats['churn_rate'] = None
    
    return stats


def find_target_column(df):
    """
    Automatically find the target (churn) column in the dataset.
    
    Parameters:
        df (pd.DataFrame): The dataset.
        
    Returns:
        str or None: The name of the target column, or None if not found.
    """
    for col in df.columns:
        col_lower = col.lower().replace('_', '').replace('.', '').strip()
        if col_lower == 'churn':
            return col
    return None
