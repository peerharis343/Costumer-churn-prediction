"""
evaluate.py - Model Evaluation Module

This module calculates evaluation metrics, compares models,
and generates visualization plots for model performance.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a single model on test data and return all metrics.
    
    Parameters:
        model: Trained sklearn model with predict and predict_proba methods.
        X_test: Test features (preprocessed).
        y_test: Test target values.
        model_name (str): Name of the model for display.
        
    Returns:
        dict: Dictionary containing all evaluation metrics.
    """
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Get probability predictions (for ROC-AUC)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        # Model doesn't support predict_proba
        y_prob = model.decision_function(X_test)
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1_score': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc': round(roc_auc_score(y_test, y_prob), 4),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'y_pred': y_pred,
        'y_prob': y_prob,
    }
    
    return metrics


def compare_models(models_dict, X_test, y_test):
    """
    Evaluate and compare all models, returning a comparison DataFrame.
    
    Parameters:
        models_dict (dict): Dictionary of {model_name: trained_model}.
        X_test: Test features (preprocessed).
        y_test: Test target values.
        
    Returns:
        tuple: (comparison_df, all_metrics_dict)
            - comparison_df: DataFrame with metrics for each model.
            - all_metrics_dict: Full metrics dictionary for each model.
    """
    all_metrics = {}
    rows = []
    
    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics[name] = metrics
        rows.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc'],
        })
    
    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.set_index('Model')
    
    return comparison_df, all_metrics


def select_best_model(comparison_df, models_dict, primary_metric='F1-Score',
                       secondary_metric='ROC-AUC', tolerance=0.01):
    """
    Select the best model based on evaluation metrics.
    
    The selection logic:
    1. Rank models by the primary metric (F1-Score by default).
    2. If top models are within 'tolerance' of each other, use secondary metric.
    
    Parameters:
        comparison_df (pd.DataFrame): Model comparison DataFrame.
        models_dict (dict): Dictionary of {model_name: trained_model}.
        primary_metric (str): Primary metric for selection.
        secondary_metric (str): Tiebreaker metric.
        tolerance (float): Tolerance for considering scores "equal".
        
    Returns:
        tuple: (best_model_name, best_model, selection_reason)
    """
    # Sort by primary metric (descending)
    sorted_df = comparison_df.sort_values(by=primary_metric, ascending=False)
    
    best_name = sorted_df.index[0]
    best_score = sorted_df[primary_metric].iloc[0]
    
    # Check if there's a close second
    reason = f"Selected based on highest {primary_metric}: {best_score:.4f}"
    
    if len(sorted_df) > 1:
        second_score = sorted_df[primary_metric].iloc[1]
        if abs(best_score - second_score) <= tolerance:
            # Use secondary metric as tiebreaker
            tied_models = sorted_df[
                abs(sorted_df[primary_metric] - best_score) <= tolerance
            ]
            tied_sorted = tied_models.sort_values(
                by=secondary_metric, ascending=False
            )
            best_name = tied_sorted.index[0]
            reason = (
                f"Models were close on {primary_metric} "
                f"(within {tolerance}). Selected based on "
                f"{secondary_metric}: {tied_sorted[secondary_metric].iloc[0]:.4f}"
            )
    
    best_model = models_dict[best_name]
    
    return best_name, best_model, reason


def plot_confusion_matrix(y_test, y_pred, model_name="Model"):
    """
    Create a confusion matrix visualization with clean spacing and readable labels.
    
    Parameters:
        y_test: True labels.
        y_pred: Predicted labels.
        model_name (str): Model name for the title.
        
    Returns:
        matplotlib.figure.Figure: The confusion matrix plot.
    """
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    
    # Create heatmap manually
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    
    # Labels
    classes = ['No Churn (0)', 'Churn (1)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel('Predicted Label', fontsize=10, fontweight='normal', labelpad=8)
    ax.set_ylabel('True Label', fontsize=10, fontweight='normal', labelpad=8)
    ax.set_title(f'Confusion Matrix\n{model_name}', fontsize=11, fontweight='bold', pad=12)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout(pad=1.5)
    return fig


def plot_roc_curves(all_metrics, y_test):
    """
    Plot ROC curves for all models on a single figure.
    
    Parameters:
        all_metrics (dict): Dictionary of metrics for each model.
        y_test: True labels.
        
    Returns:
        matplotlib.figure.Figure: The ROC curves plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#2196F3', '#4CAF50', '#FF9800']  # Blue, Green, Orange
    
    for idx, (name, metrics) in enumerate(all_metrics.items()):
        y_prob = metrics['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = metrics['roc_auc']
        
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{name} (AUC = {auc:.4f})')
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.5)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def get_metric_explanations():
    """
    Return simple English explanations of evaluation metrics.
    
    Returns:
        dict: Metric name to explanation mapping.
    """
    return {
        'Accuracy': (
            "The percentage of all predictions that were correct. "
            "For example, 90% accuracy means the model correctly predicted "
            "9 out of every 10 customers."
        ),
        'Precision': (
            "Out of all customers the model predicted would churn, "
            "what percentage actually churned. High precision means "
            "fewer false alarms (customers flagged as churning who actually stayed)."
        ),
        'Recall': (
            "Out of all customers who actually churned, what percentage "
            "the model correctly identified. High recall means the model "
            "catches most of the customers who are about to leave."
        ),
        'F1-Score': (
            "The harmonic mean of Precision and Recall. It provides a single "
            "score that balances both metrics. A high F1-Score means the model "
            "is good at both identifying churners and avoiding false alarms."
        ),
        'ROC-AUC': (
            "Area Under the ROC Curve. It measures the model's ability to "
            "distinguish between churners and non-churners across all "
            "probability thresholds. A score of 1.0 is perfect, and 0.5 "
            "is no better than random guessing."
        ),
    }
