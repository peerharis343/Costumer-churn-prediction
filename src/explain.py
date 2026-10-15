"""
explain.py - SHAP Explainability Module

This module uses SHAP (SHapley Additive exPlanations) to explain
model predictions. SHAP is an explainability technique — it does NOT
predict churn itself. Instead, it explains WHY a model made a
particular prediction by showing the contribution of each feature.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit


def get_shap_explainer(model, X_train, model_name="Model"):
    """
    Create an appropriate SHAP explainer based on the model type.
    
    Parameters:
        model: Trained sklearn/xgboost model.
        X_train: Training features (preprocessed, as numpy array or DataFrame).
        model_name (str): Name of the model (used to select explainer type).
        
    Returns:
        shap.Explainer: A SHAP explainer object.
    """
    model_name_lower = model_name.lower()
    
    if 'xgboost' in model_name_lower or 'xgb' in model_name_lower:
        # Use TreeExplainer for XGBoost (fast and exact)
        explainer = shap.TreeExplainer(model)
    elif 'random forest' in model_name_lower or 'randomforest' in model_name_lower:
        # Use TreeExplainer for Random Forest (fast and exact)
        explainer = shap.TreeExplainer(model)
    elif 'logistic' in model_name_lower:
        # Use LinearExplainer for Logistic Regression
        # Need to use a sample of training data as background
        if hasattr(X_train, 'values'):
            background = X_train.values
        else:
            background = X_train
        explainer = shap.LinearExplainer(model, background)
    else:
        # Fallback: use KernelExplainer (model-agnostic but slower)
        if hasattr(X_train, 'values'):
            background = shap.sample(X_train, min(100, len(X_train)))
        else:
            background = shap.sample(
                pd.DataFrame(X_train), min(100, len(X_train))
            ).values
        explainer = shap.KernelExplainer(model.predict_proba, background)
    
    return explainer


def get_shap_values(explainer, X_data):
    """
    Calculate SHAP values for the given data.
    
    Parameters:
        explainer: SHAP explainer object.
        X_data: Features to explain (preprocessed).
        
    Returns:
        numpy.ndarray: SHAP values array.
    """
    shap_values = explainer.shap_values(X_data)
    
    # For binary classification, shap_values might be a list [class0, class1]
    # or a 3D array (samples, features, classes)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3 and shap_values.shape[2] == 2:
        shap_values = shap_values[:, :, 1]
    elif shap_values.ndim == 2 and hasattr(X_data, 'shape') and shap_values.shape[1] == 2 and X_data.shape[1] != 2:
        shap_values = shap_values[:, 1]
    
    return shap_values


def get_feature_importance(explainer, X_data, feature_names):
    """
    Calculate global feature importance using mean absolute SHAP values.
    
    Parameters:
        explainer: SHAP explainer object.
        X_data: Test features (preprocessed).
        feature_names (list): Feature names.
        
    Returns:
        pd.DataFrame: Feature importance DataFrame sorted by importance.
    """
    shap_values = get_shap_values(explainer, X_data)
    
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = mean_abs_shap.squeeze()
    
    importance_df = pd.DataFrame({
        'Feature': feature_names[:len(mean_abs_shap)],
        'Mean |SHAP Value|': mean_abs_shap
    }).sort_values('Mean |SHAP Value|', ascending=False).reset_index(drop=True)
    
    return importance_df


def plot_shap_summary(explainer, X_data, feature_names, max_display=15):
    """
    Create a SHAP summary bar plot showing overall feature importance.
    
    Parameters:
        explainer: SHAP explainer object.
        X_data: Test features (preprocessed).
        feature_names (list): Feature names.
        max_display (int): Maximum number of features to display.
        
    Returns:
        matplotlib.figure.Figure: The SHAP summary plot.
    """
    shap_values = get_shap_values(explainer, X_data)
    
    # Create bar plot of mean |SHAP values|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = mean_abs_shap.squeeze()
    
    # Get top features
    n_features = min(max_display, len(feature_names), len(mean_abs_shap))
    indices = np.argsort(mean_abs_shap)[-n_features:]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [feature_names[i] if i < len(feature_names) else f"Feature {i}" 
             for i in indices]
    values = mean_abs_shap[indices]
    
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(names)))
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title('SHAP Feature Importance', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def explain_single_prediction(explainer, instance, feature_names, 
                                prediction_prob, base_value=None):
    """
    Explain a single prediction using SHAP values.
    Returns the top contributing features and their effects.
    
    Parameters:
        explainer: SHAP explainer object.
        instance: Single instance features (1D array or DataFrame row).
        feature_names (list): Feature names.
        prediction_prob (float): The predicted churn probability.
        base_value: The base (expected) value from the explainer.
        
    Returns:
        dict: Explanation dictionary with 'top_positive' (features pushing
              toward churn), 'top_negative' (features pushing away from churn),
              and 'shap_values' (all SHAP values for this instance).
    """
    # Reshape instance if needed
    if hasattr(instance, 'values'):
        instance_array = instance.values.reshape(1, -1)
    elif instance.ndim == 1:
        instance_array = instance.reshape(1, -1)
    else:
        instance_array = instance
    
    # Get SHAP values for this instance
    shap_values = get_shap_values(explainer, instance_array)
    shap_values = np.asarray(shap_values).squeeze()
    if shap_values.ndim > 1:
        shap_values = shap_values[0]
    
    # Create feature-value pairs
    n_features = min(len(feature_names), len(shap_values))
    feature_effects = []
    for i in range(n_features):
        s_val = float(np.ravel(shap_values[i])[0])
        f_val = float(np.ravel(instance_array)[i]) if i < instance_array.size else None
        feature_effects.append({
            'feature': feature_names[i],
            'shap_value': s_val,
            'feature_value': f_val,
            'effect': 'increases churn risk' if s_val > 0 else 'decreases churn risk'
        })
    
    # Sort by absolute SHAP value
    feature_effects.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    
    # Separate positive and negative effects
    top_positive = [f for f in feature_effects if f['shap_value'] > 0][:5]
    top_negative = [f for f in feature_effects if f['shap_value'] < 0][:5]
    
    return {
        'shap_values': shap_values,
        'feature_effects': feature_effects,
        'top_positive': top_positive,  # Features increasing churn probability
        'top_negative': top_negative,  # Features decreasing churn probability
        'prediction_prob': prediction_prob,
    }


def plot_shap_waterfall(shap_values, feature_names, base_value, instance_values=None,
                         max_display=10):
    """
    Create a waterfall plot showing how each feature contributes to a 
    single prediction.
    
    Parameters:
        shap_values: SHAP values for a single instance.
        feature_names (list): Feature names.
        base_value: Base (expected) value.
        instance_values: Original feature values for the instance.
        max_display (int): Maximum features to display.
        
    Returns:
        matplotlib.figure.Figure: The waterfall plot.
    """
    shap_values = np.asarray(shap_values).squeeze()
    if shap_values.ndim > 1:
        shap_values = shap_values[0]
    
    n_features = min(len(feature_names), len(shap_values), max_display)
    
    # Get top features by absolute SHAP value
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[-n_features:]
    
    names = [feature_names[i] if i < len(feature_names) else f"Feature {i}" 
             for i in top_indices]
    values = [float(np.ravel(shap_values[i])[0]) for i in top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FF6B6B' if v > 0 else '#4ECDC4' for v in values]
    
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('SHAP Value (impact on churn prediction)', fontsize=12)
    ax.set_title('Feature Contributions to This Prediction', fontsize=14)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='Increases churn risk'),
        Patch(facecolor='#4ECDC4', label='Decreases churn risk')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    return fig

    
    n_features = min(len(feature_names), len(shap_values), max_display)
    
    # Get top features by absolute SHAP value
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[-n_features:]
    
    names = [feature_names[i] if i < len(feature_names) else f"Feature {i}" 
             for i in top_indices]
    values = [shap_values[i] for i in top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FF6B6B' if v > 0 else '#4ECDC4' for v in values]
    
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('SHAP Value (impact on churn prediction)', fontsize=12)
    ax.set_title('Feature Contributions to This Prediction', fontsize=14)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='Increases churn risk'),
        Patch(facecolor='#4ECDC4', label='Decreases churn risk')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    return fig
