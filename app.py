"""
app.py — Customer Churn Prediction Dashboard
Pakistani Telecom Dataset

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import json
import os
import joblib

# Project modules
from src.data_loader import load_dataset, validate_dataset, get_dataset_stats
from src.preprocessing import (
    clean_data, get_feature_columns, prepare_features_target,
    load_preprocessing_artifacts, find_target_in_df
)
from src.evaluate import (
    plot_confusion_matrix, plot_roc_curves, get_metric_explanations
)
from src.explain import (
    get_shap_explainer, get_feature_importance,
    plot_shap_summary, explain_single_prediction, plot_shap_waterfall
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction - Pakistani Telecom",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Design System — Centralized CSS
# Design tokens:
#   App bg:          #202638
#   Card bg:         #171F2D
#   Elevated:        #252F43
#   Accent (green):  #35D6A0
#   Accent hover:    #29B98A
#   Secondary (purple): #655681
#   Primary text:    #F1F3F7
#   Secondary text:  #AEB8CA
#   Border:          #354055
#   Risk low:        #35D6A0
#   Risk medium:     #E8A838
#   Risk high:       #E05252
# ============================================================

st.markdown("""
<style>
    /* ---- App canvas ---- */
    .stApp {
        background-color: #202638 !important;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background-color: #171F2D !important;
        border-right: 1px solid #354055 !important;
    }
    [data-testid="stSidebar"] * {
        color: #AEB8CA !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] strong {
        color: #F1F3F7 !important;
    }

    /* ---- Sidebar radio nav ---- */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 4px !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {
        display: flex !important;
        align-items: center !important;
        min-height: 38px !important;
        padding: 7px 12px !important;
        border-radius: 6px !important;
        background-color: transparent !important;
        border: 1px solid #354055 !important;
        margin-bottom: 3px !important;
        cursor: pointer !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        background-color: #252F43 !important;
        border-color: #655681 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        background-color: #252F43 !important;
        border-color: #35D6A0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] span {
        color: #AEB8CA !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) span {
        color: #35D6A0 !important;
        font-weight: 600 !important;
    }

    /* ---- Page headings ---- */
    .page-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F1F3F7 !important;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .page-subtitle {
        font-size: 0.9rem;
        color: #AEB8CA !important;
        margin-bottom: 1.25rem;
    }

    /* ---- Section headers ---- */
    .section-header {
        border-left: 3px solid #35D6A0;
        padding-left: 12px;
        margin: 1.75rem 0 1rem 0;
    }
    .section-header h2, .section-header h3 {
        margin: 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #F1F3F7 !important;
        letter-spacing: -0.01em;
    }

    /* ---- KPI metric cards ---- */
    .metric-card {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
        min-height: 110px;
        padding: 1.1rem 1.25rem;
        border-radius: 8px;
        background-color: #171F2D;
        border: 1px solid #354055;
        border-top: 3px solid var(--card-accent, #35D6A0);
        margin-bottom: 1rem;
        box-sizing: border-box;
    }
    .metric-card .metric-label {
        color: #AEB8CA !important;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #F1F3F7 !important;
        line-height: 1.1;
    }
    .metric-card .subtext {
        font-size: 0.78rem;
        color: #AEB8CA !important;
        margin-top: 0.3rem;
    }

    /* ---- Risk badges ---- */
    .risk-low {
        background-color: rgba(53, 214, 160, 0.15);
        color: #35D6A0 !important;
        border: 1px solid #35D6A0;
        padding: 5px 16px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    .risk-medium {
        background-color: rgba(232, 168, 56, 0.15);
        color: #E8A838 !important;
        border: 1px solid #E8A838;
        padding: 5px 16px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    .risk-high {
        background-color: rgba(224, 82, 82, 0.15);
        color: #E05252 !important;
        border: 1px solid #E05252;
        padding: 5px 16px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }

    /* ---- Prediction result panels ---- */
    .prediction-box {
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        margin: 1.25rem 0;
        border: 1px solid #354055;
    }
    .prediction-churn {
        background-color: #271A1A;
        border-left: 4px solid #E05252;
        color: #F1F3F7 !important;
    }
    .prediction-churn h3, .prediction-churn p {
        color: #F1F3F7 !important;
    }
    .prediction-no-churn {
        background-color: #132820;
        border-left: 4px solid #35D6A0;
        color: #F1F3F7 !important;
    }
    .prediction-no-churn h3, .prediction-no-churn p {
        color: #F1F3F7 !important;
    }

    /* ---- Info / Note boxes ---- */
    .info-box {
        background-color: #171F2D;
        color: #AEB8CA !important;
        border-left: 3px solid #655681;
        padding: 1rem 1.1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.9rem 0;
        border-top: 1px solid #354055;
        border-right: 1px solid #354055;
        border-bottom: 1px solid #354055;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .info-box strong {
        color: #35D6A0 !important;
    }

    /* ---- Best model highlight ---- */
    .best-model-box {
        background-color: #252F43;
        border: 1px solid #354055;
        border-top: 3px solid #35D6A0;
        color: #F1F3F7 !important;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .best-model-box h3 {
        color: #35D6A0 !important;
        font-size: 1.1rem;
        margin-bottom: 0.4rem;
    }
    .best-model-box p {
        color: #AEB8CA !important;
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.5;
    }

    /* ---- Primary buttons (form submit) ---- */
    [data-testid="stFormSubmitButton"] button,
    .stButton button[kind="primary"] {
        background-color: #35D6A0 !important;
        color: #0D1117 !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.5rem !important;
        min-height: 44px !important;
        border: 1px solid #35D6A0 !important;
        border-radius: 6px !important;
        transition: background-color 0.15s ease !important;
        box-shadow: none !important;
    }
    [data-testid="stFormSubmitButton"] button:hover,
    .stButton button[kind="primary"]:hover {
        background-color: #29B98A !important;
        border-color: #29B98A !important;
    }

    /* ---- Download / secondary buttons ---- */
    .stDownloadButton button {
        background-color: #252F43 !important;
        color: #F1F3F7 !important;
        border: 1px solid #354055 !important;
        min-height: 40px !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
        box-shadow: none !important;
    }
    .stDownloadButton button:hover {
        background-color: #354055 !important;
        border-color: #655681 !important;
        color: #F1F3F7 !important;
    }

    /* ---- Expanders ---- */
    [data-testid="stExpander"] {
        border: 1px solid #354055 !important;
        border-radius: 6px !important;
        background-color: #171F2D !important;
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stExpander"]:hover {
        border-color: #655681 !important;
    }
    [data-testid="stExpander"][open] {
        border-color: #35D6A0 !important;
    }
    [data-testid="stExpander"] summary {
        padding: 0.7rem 1rem !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #F1F3F7 !important;
        cursor: pointer !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: #35D6A0 !important;
    }
    [data-testid="stExpander"] summary svg {
        color: #AEB8CA !important;
        fill: #AEB8CA !important;
    }
    [data-testid="stExpander"][open] summary {
        border-bottom: 1px solid #354055 !important;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: 0.9rem 1rem !important;
        color: #AEB8CA !important;
        font-size: 0.9rem !important;
        line-height: 1.6 !important;
    }

    /* ---- Form inputs & selects ---- */
    .stSelectbox label,
    .stNumberInput label,
    .stTextInput label,
    .stMultiSelect label,
    .stSlider label,
    .stFileUploader label {
        color: #AEB8CA !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {
        background-color: #171F2D !important;
        border-color: #354055 !important;
        color: #F1F3F7 !important;
    }
    [data-baseweb="select"] [data-baseweb="tag"] {
        background-color: #252F43 !important;
        color: #F1F3F7 !important;
    }
    .stTextInput input {
        background-color: #171F2D !important;
        color: #F1F3F7 !important;
        border-color: #354055 !important;
    }
    .stNumberInput input {
        background-color: #171F2D !important;
        color: #F1F3F7 !important;
        border-color: #354055 !important;
    }

    /* ---- Streamlit alert boxes ---- */
    [data-testid="stAlert"] {
        background-color: #171F2D !important;
        border-color: #354055 !important;
        color: #AEB8CA !important;
        border-radius: 6px !important;
    }
    [data-testid="stAlert"] p {
        color: #AEB8CA !important;
    }
    div[data-baseweb="notification"] {
        background-color: #171F2D !important;
        border-color: #354055 !important;
        color: #AEB8CA !important;
    }

    /* ---- Dataframes / tables ---- */
    [data-testid="stDataFrame"] {
        background-color: #171F2D !important;
        border: 1px solid #354055 !important;
        border-radius: 6px !important;
    }
    .dvn-scroller {
        background-color: #171F2D !important;
    }

    /* ---- File uploader ---- */
    [data-testid="stFileUploadDropzone"] {
        background-color: #171F2D !important;
        border: 1px dashed #354055 !important;
        border-radius: 6px !important;
        color: #AEB8CA !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #35D6A0 !important;
    }

    /* ---- Markdown default text ---- */
    .stMarkdown p, .stMarkdown li {
        color: #AEB8CA !important;
        font-size: 0.9rem !important;
    }
    .stMarkdown strong {
        color: #F1F3F7 !important;
    }

    /* ---- Captions ---- */
    [data-testid="stCaptionContainer"] p {
        color: #AEB8CA !important;
        font-size: 0.8rem !important;
    }

    /* ---- st.metric boxes ---- */
    [data-testid="stMetricValue"] {
        color: #F1F3F7 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #AEB8CA !important;
    }

    /* ---- Horizontal dividers ---- */
    hr {
        border-color: #354055 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Cached Data Loading
# ============================================================

@st.cache_data
def load_raw_data():
    """Load the raw dataset."""
    return load_dataset('data/churnData.csv')

@st.cache_data
def load_clean_data():
    """Load and clean the dataset."""
    df = load_raw_data()
    return clean_data(df)

@st.cache_resource
def load_trained_models():
    """Load trained models and preprocessing artifacts."""
    models = joblib.load('models/all_models.pkl')
    best_model = joblib.load('models/best_model.pkl')
    preprocessing = load_preprocessing_artifacts('models')
    test_data = joblib.load('models/test_data.pkl')

    with open('models/evaluation_results.json', 'r') as f:
        eval_results = json.load(f)

    return models, best_model, preprocessing, test_data, eval_results


# ============================================================
# Risk Level Classification
# ============================================================

def get_risk_level(probability):
    """
    Classify churn probability into a risk tier.

    Thresholds:
        Low Risk:    probability < 30%
        Medium Risk: 30% <= probability < 70%
        High Risk:   probability >= 70%
    """
    if probability < 0.30:
        return "Low Risk", "risk-low"
    elif probability < 0.70:
        return "Medium Risk", "risk-medium"
    else:
        return "High Risk", "risk-high"


@st.cache_data
def get_dataset_predictions():
    """
    Generate predictions across the full dataset.
    Cached so results are computed once and shared between Dashboard and
    High-Risk Customers pages.
    """
    df_raw = load_raw_data()
    models, best_model, preprocessing, test_data, eval_results = load_trained_models()
    preprocessor = preprocessing['preprocessor']

    df_cleaned = clean_data(df_raw, drop_duplicates=False)
    X_processed = preprocessor.transform(df_cleaned)

    preds = best_model.predict(X_processed)
    probs = best_model.predict_proba(X_processed)[:, 1]

    results_df = df_raw.copy()
    results_df.insert(0, 'Customer Reference', [f"Customer #{i+1}" for i in range(len(results_df))])
    results_df['Churn Prediction'] = ["Yes" if p == 1 else "No" for p in preds]
    results_df['Churn Probability (%)'] = (probs * 100).round(1)
    results_df['Churn_Prob_Raw'] = probs
    results_df['Risk Level'] = [get_risk_level(p)[0] for p in probs]

    return results_df, eval_results


# ============================================================
# Plotly Layout Helper
# ============================================================

def plotly_layout_defaults(title="", height=380, legend=None):
    """
    Return a dict of Plotly layout kwargs consistent with the dark theme.
    Apply with fig.update_layout(**plotly_layout_defaults(...)).
    """
    defaults = dict(
        title=dict(
            text=title,
            font=dict(color="#F1F3F7", size=13),
            x=0,
            xanchor="left",
            pad=dict(l=4),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=16, r=16, t=44, b=28),
        font=dict(color="#AEB8CA", size=12),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="#AEB8CA", size=11),
            title_font=dict(color="#AEB8CA"),
            linecolor="#354055",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#354055",
            zeroline=False,
            tickfont=dict(color="#AEB8CA", size=11),
            title_font=dict(color="#AEB8CA"),
            linecolor="#354055",
        ),
    )
    if legend:
        defaults["legend"] = legend
    else:
        defaults["legend"] = dict(
            font=dict(color="#AEB8CA", size=11),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#354055",
        )
    return defaults


# ============================================================
# Sidebar Navigation
# ============================================================

with st.sidebar:
    st.markdown("**Navigation**")
    page = st.radio(
        "Go to",
        [
            "Dashboard",
            "Model Evaluation",
            "Manual Prediction",
            "Batch Prediction",
            "High-Risk Customers",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
**About**

Customer churn prediction for Pakistani telecom networks using machine learning and SHAP explainability.

**Dataset:** Pakistan Telecom (Kaggle)
**Models:** Logistic Regression, Random Forest, XGBoost
**Explainability:** SHAP
""")

    st.markdown("---")
    st.markdown("""
**Risk thresholds**

Low Risk: below 30%
Medium Risk: 30% – 70%
High Risk: 70% and above
""")


# ============================================================
# PAGE 1: Dashboard
# ============================================================

if page == "Dashboard":
    st.markdown('<p class="page-title">Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Churn prediction summary across the full subscriber dataset.</p>', unsafe_allow_html=True)

    try:
        results_df, eval_results = get_dataset_predictions()
        total_customers = len(results_df)

        pred_churners = int((results_df['Churn Prediction'] == 'Yes').sum())
        pred_churn_rate = (pred_churners / total_customers) * 100

        high_risk_count = int((results_df['Churn_Prob_Raw'] >= 0.70).sum())
        high_risk_pct = (high_risk_count / total_customers) * 100

        best_model_name = eval_results.get('best_model_name', 'Random Forest')
        best_roc_auc = next(
            (m['ROC-AUC'] for m in eval_results.get('comparison', []) if m.get('Model') == best_model_name),
            None
        )
        roc_auc_display = f"{best_roc_auc:.4f}" if best_roc_auc is not None else "—"

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card" style="--card-accent: #35D6A0;">
                <div class="metric-label">Total customers</div>
                <div class="value">{total_customers:,}</div>
                <div class="subtext">Subscriber records</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card" style="--card-accent: #E05252;">
                <div class="metric-label">Predicted churn rate</div>
                <div class="value">{pred_churn_rate:.1f}%</div>
                <div class="subtext">{pred_churners:,} predicted churners</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card" style="--card-accent: #E8A838;">
                <div class="metric-label">High-risk customers</div>
                <div class="value">{high_risk_count:,}</div>
                <div class="subtext">{high_risk_pct:.1f}% with churn risk ≥ 70%</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card" style="--card-accent: #655681;">
                <div class="metric-label">Best model ROC-AUC</div>
                <div class="value">{roc_auc_display}</div>
                <div class="subtext">{best_model_name}</div>
            </div>
            """, unsafe_allow_html=True)

        # Row 1: Risk distribution + Network churn
        col_risk, col_net = st.columns(2)

        with col_risk:
            st.markdown('<div class="section-header"><h2>Risk distribution</h2></div>', unsafe_allow_html=True)

            risk_counts = results_df['Risk Level'].value_counts()
            risk_categories = ["Low Risk", "Medium Risk", "High Risk"]
            risk_chart_data = []
            for r in risk_categories:
                cnt = int(risk_counts.get(r, 0))
                pct = (cnt / total_customers) * 100
                risk_chart_data.append({
                    'Risk Category': r,
                    'Subscribers': cnt,
                    'Percentage': pct,
                    'Label': f"{cnt:,} ({pct:.1f}%)"
                })
            risk_chart_df = pd.DataFrame(risk_chart_data)

            fig_risk = px.bar(
                risk_chart_df,
                x='Risk Category',
                y='Subscribers',
                text='Label',
                color='Risk Category',
                color_discrete_map={
                    'Low Risk': '#35D6A0',
                    'Medium Risk': '#E8A838',
                    'High Risk': '#E05252'
                },
            )
            fig_risk.update_traces(
                textposition='outside',
                hovertemplate="<b>%{x}</b><br>Subscribers: %{y:,}<extra></extra>"
            )
            layout = plotly_layout_defaults("Subscribers by risk tier", height=360)
            layout['showlegend'] = False
            layout['yaxis']['range'] = [0, max(risk_chart_df['Subscribers']) * 1.22]
            layout['yaxis']['title'] = "Subscribers"
            fig_risk.update_layout(**layout)
            st.plotly_chart(fig_risk, use_container_width=True)
            st.caption("Low: < 30% probability · Medium: 30–70% · High: ≥ 70%")

        with col_net:
            st.markdown('<div class="section-header"><h2>Network churn comparison</h2></div>', unsafe_allow_html=True)

            net_stats = results_df.groupby('Network', dropna=False).agg(
                Total=('Churn_Prob_Raw', 'count'),
                Churners=('Churn Prediction', lambda s: (s == 'Yes').sum()),
                Churn_Rate=('Churn Prediction', lambda s: (s == 'Yes').mean() * 100)
            ).reset_index()
            net_stats['Network'] = net_stats['Network'].fillna('Unknown')
            net_stats = net_stats.sort_values(by='Churn_Rate', ascending=False)
            net_stats['Display_Label'] = net_stats.apply(
                lambda r: f"{r['Churn_Rate']:.1f}% ({r['Churners']:,}/{r['Total']:,})", axis=1
            )

            fig_net = px.bar(
                net_stats,
                x='Network',
                y='Churn_Rate',
                text='Display_Label',
                color='Churn_Rate',
                color_continuous_scale=['#35D6A0', '#655681', '#E05252'],
            )
            fig_net.update_traces(
                textposition='outside',
                hovertemplate="<b>%{x}</b><br>Predicted churn rate: %{y:.1f}%<extra></extra>"
            )
            layout_net = plotly_layout_defaults("Predicted churn rate by network provider", height=360)
            layout_net['coloraxis_showscale'] = False
            layout_net['yaxis']['range'] = [0, max(net_stats['Churn_Rate']) * 1.28]
            layout_net['yaxis']['title'] = "Churn rate (%)"
            fig_net.update_layout(**layout_net)
            st.plotly_chart(fig_net, use_container_width=True)
            st.caption("Churner counts shown as churners / total per provider.")

        # Row 2: Satisfaction + High-risk preview
        col_sat, col_high = st.columns(2)

        with col_sat:
            st.markdown('<div class="section-header"><h2>Churn rate by satisfaction factor</h2></div>', unsafe_allow_html=True)

            sat_mapping = {
                'satisfy_with_voice_quality': 'Voice quality',
                'Happy_with_intrnet_quality': 'Internet quality',
                'satisfy_with_signal_strength': 'Signal strength',
                'satisfy_with_call_rate': 'Call rates',
                'pleaseed_with_prices': 'Prices',
                'satisfy_with_billing_process': 'Billing'
            }

            sat_rows = []
            for col_key, display_name in sat_mapping.items():
                if col_key in results_df.columns:
                    no_mask = results_df[col_key].astype(str).str.strip().str.lower() == 'no'
                    yes_mask = results_df[col_key].astype(str).str.strip().str.lower() == 'yes'
                    no_rate = (results_df.loc[no_mask, 'Churn Prediction'] == 'Yes').mean() * 100 if no_mask.sum() > 0 else 0
                    yes_rate = (results_df.loc[yes_mask, 'Churn Prediction'] == 'Yes').mean() * 100 if yes_mask.sum() > 0 else 0
                    sat_rows.append({'Attribute': display_name, 'Group': 'Dissatisfied', 'Churn Rate (%)': round(no_rate, 1)})
                    sat_rows.append({'Attribute': display_name, 'Group': 'Satisfied', 'Churn Rate (%)': round(yes_rate, 1)})

            sat_df = pd.DataFrame(sat_rows)

            fig_sat = px.bar(
                sat_df,
                x='Attribute',
                y='Churn Rate (%)',
                color='Group',
                barmode='group',
                color_discrete_map={
                    'Dissatisfied': '#E05252',
                    'Satisfied': '#35D6A0'
                },
            )
            fig_sat.update_traces(
                hovertemplate="<b>%{x}</b> (%{data.name})<br>Churn rate: %{y:.1f}%<extra></extra>"
            )
            layout_sat = plotly_layout_defaults("Predicted churn rate by satisfaction", height=370)
            layout_sat['yaxis']['title'] = "Churn rate (%)"
            layout_sat['legend'] = dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                font=dict(color="#AEB8CA", size=11),
                bgcolor="rgba(0,0,0,0)",
            )
            fig_sat.update_layout(**layout_sat)
            st.plotly_chart(fig_sat, use_container_width=True)
            st.caption("Correlation between satisfaction responses and predicted churn. Not causal.")

        with col_high:
            st.markdown('<div class="section-header"><h2>Top high-risk customers</h2></div>', unsafe_allow_html=True)

            top_risk_df = results_df.sort_values(by='Churn_Prob_Raw', ascending=False).head(8)
            preview_display_cols = ['Customer Reference', 'Network', 'service_type', 'Churn Probability (%)', 'Risk Level']

            st.dataframe(
                top_risk_df[preview_display_cols],
                use_container_width=True,
                hide_index=True
            )
            st.info(f"Full list of {high_risk_count:,} high-risk subscribers is available on the High-Risk Customers page.")

        # Business Insights
        st.markdown('<div class="section-header"><h2>Dataset observations</h2></div>', unsafe_allow_html=True)

        bcol1, bcol2, bcol3 = st.columns(3)

        with bcol1:
            if len(net_stats) > 0:
                top_net_row = net_stats.iloc[0]
                top_net_name = top_net_row['Network']
                top_net_rate = top_net_row['Churn_Rate']
                top_net_churners = int(top_net_row['Churners'])
                top_net_total = int(top_net_row['Total'])
                low_net_row = net_stats.iloc[-1]
                low_net_name = low_net_row['Network']
                low_net_rate = low_net_row['Churn_Rate']
                low_net_churners = int(low_net_row['Churners'])
                low_net_total = int(low_net_row['Total'])
                network_insight = (
                    f"<strong>{top_net_name}</strong> has the highest predicted churn rate at "
                    f"<strong>{top_net_rate:.1f}%</strong> ({top_net_churners:,} of {top_net_total:,}). "
                    f"<strong>{low_net_name}</strong> has the lowest at "
                    f"<strong>{low_net_rate:.1f}%</strong> ({low_net_churners:,} of {low_net_total:,})."
                )
            else:
                network_insight = "Insufficient network data."
            st.markdown(f"""
            <div class="info-box">
                <strong>Network disparity</strong><br>
                {network_insight}
            </div>
            """, unsafe_allow_html=True)

        with bcol2:
            low_risk_cnt = int(risk_counts.get('Low Risk', 0))
            med_risk_cnt = int(risk_counts.get('Medium Risk', 0))
            high_risk_cnt = int(risk_counts.get('High Risk', 0))
            low_risk_pct_val = (low_risk_cnt / total_customers) * 100
            high_risk_pct_val = (high_risk_cnt / total_customers) * 100
            st.markdown(f"""
            <div class="info-box">
                <strong>Risk tier distribution</strong><br>
                <strong>{low_risk_cnt:,}</strong> customers ({low_risk_pct_val:.1f}%) are in the Low Risk tier.
                <strong>{high_risk_cnt:,}</strong> customers ({high_risk_pct_val:.1f}%) are in the High Risk tier
                and represent the primary retention priority.
            </div>
            """, unsafe_allow_html=True)

        with bcol3:
            if len(sat_df) > 0:
                dissatisfied_df = sat_df[sat_df['Group'] == 'Dissatisfied'].copy()
                satisfied_df = sat_df[sat_df['Group'] == 'Satisfied'].copy()
                if len(dissatisfied_df) > 0:
                    worst_sat_row = dissatisfied_df.sort_values('Churn Rate (%)', ascending=False).iloc[0]
                    worst_attr = worst_sat_row['Attribute']
                    worst_dis_rate = worst_sat_row['Churn Rate (%)']
                    sat_match = satisfied_df[satisfied_df['Attribute'] == worst_attr]
                    worst_sat_rate = sat_match['Churn Rate (%)'].values[0] if len(sat_match) > 0 else 0
                    ratio_txt = f"{worst_dis_rate/worst_sat_rate:.1f}x" if worst_sat_rate > 0 else "higher"
                    quality_insight = (
                        f"Dissatisfied subscribers in <strong>{worst_attr}</strong> show a predicted churn rate of "
                        f"<strong>{worst_dis_rate:.1f}%</strong> — {ratio_txt} the rate for satisfied subscribers "
                        f"({worst_sat_rate:.1f}%)."
                    )
                else:
                    quality_insight = "Insufficient satisfaction data for comparison."
            else:
                quality_insight = "Satisfaction data is unavailable."
            st.markdown(f"""
            <div class="info-box">
                <strong>Satisfaction and churn</strong><br>
                {quality_insight}
            </div>
            """, unsafe_allow_html=True)

        st.caption("Observations are based on model predictions applied to the dataset. They reflect statistical patterns, not causal relationships.")

    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Dashboard error: {e}")


# ============================================================
# PAGE 2: Model Evaluation
# ============================================================

elif page == "Model Evaluation":
    st.markdown('<p class="page-title">Model Evaluation</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Performance comparison of trained classification models on the held-out test set.</p>', unsafe_allow_html=True)

    try:
        models, best_model, preprocessing, test_data, eval_results = load_trained_models()

        X_test = test_data['X_test']
        y_test = test_data['y_test']

        # Model comparison table
        st.markdown('<div class="section-header"><h2>Model comparison</h2></div>', unsafe_allow_html=True)

        comparison_data = eval_results['comparison']
        comp_df = pd.DataFrame(comparison_data)
        if 'Model' in comp_df.columns:
            comp_df = comp_df.set_index('Model')

        comp_styled = (
            comp_df.style
            .highlight_max(axis=0, props='background-color: #1A3328; color: #35D6A0; font-weight: 600;')
            .format('{:.4f}')
        )
        st.dataframe(comp_styled, use_container_width=True)

        # Best model
        best_name = eval_results['best_model_name']
        st.markdown(f"""
        <div class="best-model-box">
            <h3>Selected model: {best_name}</h3>
            <p>{eval_results['selection_reason']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Training details
        st.markdown('<div class="section-header"><h2>Training details</h2></div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Training samples", eval_results['train_size'])
        col2.metric("Test samples", eval_results['test_size'])
        col3.metric("Features", eval_results['n_features'])

        st.info(
            f"Train/test split: {int((1-eval_results['test_ratio'])*100)}/{int(eval_results['test_ratio']*100)} "
            f"(random_state={eval_results['random_state']})"
        )

        # Metric explanations
        st.markdown('<div class="section-header"><h2>Metric explanations</h2></div>', unsafe_allow_html=True)

        explanations = get_metric_explanations()
        for metric_name, explanation in explanations.items():
            with st.expander(metric_name):
                st.write(explanation)

        # Confusion matrices
        st.markdown('<div class="section-header"><h2>Confusion matrices</h2></div>', unsafe_allow_html=True)

        cols = st.columns(3, gap="large")
        for idx, (name, model) in enumerate(models.items()):
            with cols[idx]:
                y_pred = model.predict(X_test)
                fig = plot_confusion_matrix(y_test, y_pred, name)
                st.pyplot(fig)
                plt.close(fig)

        # ROC curves
        st.markdown('<div class="section-header"><h2>ROC curves</h2></div>', unsafe_allow_html=True)

        all_metrics = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc_val = None
            for record in eval_results['comparison']:
                if record['Model'] == name:
                    roc_auc_val = record['ROC-AUC']
                    break
            all_metrics[name] = {
                'y_pred': y_pred,
                'y_prob': y_prob,
                'roc_auc': roc_auc_val or 0
            }

        fig = plot_roc_curves(all_metrics, y_test)
        st.pyplot(fig)
        plt.close(fig)

        # SHAP feature importance
        st.markdown('<div class="section-header"><h2>SHAP feature importance</h2></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <strong>About SHAP</strong><br>
            SHAP (SHapley Additive exPlanations) quantifies each feature's contribution to individual model
            predictions. Higher absolute SHAP values indicate stronger influence on the output.
            SHAP does not predict churn — it explains how the model reached its predictions.
        </div>
        """, unsafe_allow_html=True)

        try:
            feature_names = test_data['feature_names']
            X_train = test_data['X_train']

            explainer = get_shap_explainer(best_model, X_train, best_name)

            sample_size = min(200, X_test.shape[0])
            X_sample = X_test[:sample_size]

            fig = plot_shap_summary(explainer, X_sample, feature_names)
            st.pyplot(fig)
            plt.close(fig)

            importance_df = get_feature_importance(explainer, X_sample, feature_names)
            st.dataframe(importance_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.warning(f"SHAP explanations unavailable: {e}")

    except FileNotFoundError:
        st.error("Models not found. Run `python -m src.train` to train the models first.")
    except Exception as e:
        st.error(f"Error loading models: {e}")


# ============================================================
# PAGE 3: Manual Prediction
# ============================================================

elif page == "Manual Prediction":
    st.markdown('<p class="page-title">Manual prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Enter customer details to predict churn probability for a single subscriber.</p>', unsafe_allow_html=True)

    try:
        models, best_model, preprocessing, test_data, eval_results = load_trained_models()

        preprocessor = preprocessing['preprocessor']
        numerical_cols = preprocessing['numerical_cols']
        categorical_cols = preprocessing['categorical_cols']
        feature_names = test_data['feature_names']
        best_name = eval_results['best_model_name']

        st.markdown(f"""
        <div class="info-box">
            <strong>Active model:</strong> {best_name}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header"><h2>Customer details</h2></div>', unsafe_allow_html=True)

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Demographics & network**")
                gender = st.selectbox("Gender", options=["Female", "Male"])
                network = st.selectbox("Network operator", options=["Jazz / Warid", "Zong", "Ufone", "Telenor"])
                relation_dur = st.selectbox("Relationship duration", options=["More than one year", "7 12 Months", "2 6 Months", "Less than one Month"])
                service_type = st.selectbox("Service type", options=["Pre Paid", "Post Paid"])
                source_attraction = st.selectbox("Source of attraction", options=["Advertisement", "Others", "Face book", "Newspaper"])
                pref_comm = st.selectbox("Preferred communication", options=["Audio calling", "messages", "Video Calling"])

            with col2:
                st.markdown("**Customer service & usage**")
                contact_cc = st.selectbox("Contacted customer care?", options=["No", "Yes"])
                rate_cc = st.selectbox("Customer care call frequency", options=["rarely", "Once in a year", "Once in a Month", "Once in a Week"])
                reason_cc = st.selectbox("Reason for contacting customer care", options=["Complaints", "Information regarding new schemes", "Other Queries", "Value added services"])
                monthly_usage = st.number_input("Monthly usage (Rs.)", min_value=0, max_value=10000, value=500, step=50)
                happy_internet = st.selectbox("Satisfied with internet quality?", options=["Yes", "No"])
                fg_available = st.selectbox("4G available?", options=["Yes", "No"])

            with col3:
                st.markdown("**Satisfaction ratings**")
                satisfy_call_rate = st.selectbox("Satisfied with call rate?", options=["Yes", "No"])
                satisfy_voice = st.selectbox("Satisfied with voice quality?", options=["Yes", "No"])
                satisfy_signal = st.selectbox("Satisfied with signal strength?", options=["Yes", "No"])
                pleased_prices = st.selectbox("Pleased with prices?", options=["Yes", "No"])
                satisfy_billing = st.selectbox("Satisfied with billing process?", options=["Yes", "No"])
                recommend_others = st.selectbox("Would recommend to others?", options=["Yes", "No"])

            submitted = st.form_submit_button(
                "Predict churn",
                type="primary",
                use_container_width=True
            )

        if submitted:
            try:
                raw_input_data = pd.DataFrame([{
                    'Gender': gender,
                    'Network': network,
                    'relation_durattion': relation_dur,
                    'service_type': service_type,
                    'source_of_attraction': source_attraction,
                    'prferred_communication': pref_comm,
                    'satisfy_with_call_rate': satisfy_call_rate,
                    'contact_with_cust_care': contact_cc,
                    'rate_of_cust_care_calling': rate_cc,
                    'reascon_of_contact_cc': reason_cc,
                    'monthly_usage_rs': monthly_usage,
                    'Happy_with_intrnet_quality': happy_internet,
                    '4G_available': fg_available,
                    'satisfy_with_voice_quality': satisfy_voice,
                    'satisfy_with_signal_strength': satisfy_signal,
                    'pleaseed_with_prices': pleased_prices,
                    'satisfy_with_billing_process': satisfy_billing,
                    'recommend_to_others': recommend_others
                }])

                cleaned_input = clean_data(raw_input_data)
                input_processed = preprocessor.transform(cleaned_input)

                prediction = best_model.predict(input_processed)[0]
                probability = best_model.predict_proba(input_processed)[0]

                churn_prob = float(probability[1])
                no_churn_prob = float(probability[0])

                risk_level, risk_class = get_risk_level(churn_prob)

                st.markdown("---")
                st.markdown('<div class="section-header"><h2>Prediction result</h2></div>', unsafe_allow_html=True)

                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box prediction-churn">
                        <h3>Churn predicted</h3>
                        <p style="font-size: 2rem; font-weight: 700; margin: 0.5rem 0;">
                            {churn_prob*100:.1f}%
                        </p>
                        <p><span class="{risk_class}">{risk_level}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box prediction-no-churn">
                        <h3>No churn predicted</h3>
                        <p style="font-size: 2rem; font-weight: 700; margin: 0.5rem 0;">
                            {churn_prob*100:.1f}%
                        </p>
                        <p><span class="{risk_class}">{risk_level}</span></p>
                    </div>
                    """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                col1.metric("Churn probability", f"{churn_prob*100:.1f}%")
                col2.metric("Retention probability", f"{no_churn_prob*100:.1f}%")
                col3.metric("Risk level", risk_level)

                st.markdown("""
                <div class="info-box">
                    <strong>Risk thresholds</strong><br>
                    Low Risk: below 30%<br>
                    Medium Risk: 30% to 70%<br>
                    High Risk: 70% and above
                </div>
                """, unsafe_allow_html=True)

                # SHAP explanation
                st.markdown('<div class="section-header"><h2>SHAP prediction explanation</h2></div>', unsafe_allow_html=True)

                st.markdown("""
                <div class="info-box">
                    <strong>About SHAP explanations</strong><br>
                    The waterfall chart below shows which features increased or decreased the churn probability
                    for this specific customer. SHAP values represent each feature's marginal contribution to
                    the model output relative to the baseline (average) prediction.
                </div>
                """, unsafe_allow_html=True)

                try:
                    X_train = test_data['X_train']
                    explainer = get_shap_explainer(best_model, X_train, best_name)

                    explanation = explain_single_prediction(
                        explainer, input_processed, feature_names, churn_prob
                    )

                    base_value = explainer.expected_value
                    if isinstance(base_value, (list, np.ndarray)):
                        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

                    fig = plot_shap_waterfall(
                        explanation['shap_values'], feature_names,
                        base_value, input_processed
                    )
                    st.pyplot(fig)
                    plt.close(fig)

                    st.markdown("**Top contributing features**")

                    if explanation['top_positive']:
                        st.markdown("Features increasing churn risk:")
                        for factor in explanation['top_positive'][:3]:
                            st.markdown(
                                f"- [+] **{factor['feature']}** "
                                f"(SHAP: {factor['shap_value']:.4f})"
                            )

                    if explanation['top_negative']:
                        st.markdown("Features decreasing churn risk:")
                        for factor in explanation['top_negative'][:3]:
                            st.markdown(
                                f"- [-] **{factor['feature']}** "
                                f"(SHAP: {factor['shap_value']:.4f})"
                            )

                except Exception as e:
                    st.warning(f"SHAP explanation unavailable: {e}")

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Ensure all input fields contain valid values.")

    except FileNotFoundError:
        st.error("Models not found. Run `python -m src.train` to train the models first.")
    except Exception as e:
        st.error(f"Error: {e}")


# ============================================================
# PAGE 4: Batch Prediction
# ============================================================

elif page == "Batch Prediction":
    st.markdown('<p class="page-title">Batch prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Upload a CSV file to generate churn predictions for multiple customers.</p>', unsafe_allow_html=True)

    try:
        models, best_model, preprocessing, test_data, eval_results = load_trained_models()
        preprocessor = preprocessing['preprocessor']
        best_name = eval_results['best_model_name']

        required_features = [
            'Gender', 'Network', 'relation_durattion', 'service_type',
            'source_of_attraction', 'prferred_communication', 'satisfy_with_call_rate',
            'contact_with_cust_care', 'rate_of_cust_care_calling', 'reascon_of_contact_cc',
            'monthly_usage_rs', 'Happy_with_intrnet_quality', '4G_available',
            'satisfy_with_voice_quality', 'satisfy_with_signal_strength',
            'pleaseed_with_prices', 'satisfy_with_billing_process', 'recommend_to_others'
        ]

        uploaded_file = st.file_uploader("Upload customer dataset (CSV)", type=["csv"])

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.markdown(f"**Loaded {len(raw_df):,} record(s) from uploaded file.**")

                missing_cols = [col for col in required_features if col not in raw_df.columns]

                if missing_cols:
                    st.error(
                        f"Validation error: the uploaded CSV is missing {len(missing_cols)} required column(s): "
                        + ", ".join([f"`{c}`" for c in missing_cols])
                    )
                    st.info("Ensure your CSV contains all 18 required feature columns listed in the schema below.")
                else:
                    cleaned_df = clean_data(raw_df)
                    processed_X = preprocessor.transform(cleaned_df)

                    preds = best_model.predict(processed_X)
                    probs = best_model.predict_proba(processed_X)[:, 1]

                    results_df = raw_df.copy()
                    results_df['Predicted_Churn'] = ["Yes" if p == 1 else "No" for p in preds]
                    results_df['Churn_Probability_%'] = (probs * 100).round(1)
                    results_df['Risk_Level'] = [get_risk_level(p)[0] for p in probs]

                    st.markdown('<div class="section-header"><h2>Prediction summary</h2></div>', unsafe_allow_html=True)

                    sc1, sc2, sc3, sc4 = st.columns(4)

                    total_cust = len(results_df)
                    pred_churners_b = int((preds == 1).sum())
                    pred_non_churners_b = int((preds == 0).sum())
                    high_risk_cnt_b = int(sum(p >= 0.70 for p in probs))

                    sc1.metric("Total customers", f"{total_cust:,}")
                    sc2.metric("Predicted churners", f"{pred_churners_b:,}")
                    sc3.metric("Predicted non-churners", f"{pred_non_churners_b:,}")
                    sc4.metric("High-risk (≥ 70%)", f"{high_risk_cnt_b:,}")

                    st.markdown('<div class="section-header"><h2>Results</h2></div>', unsafe_allow_html=True)
                    st.dataframe(results_df, use_container_width=True)

                    csv_bytes = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Export results CSV",
                        data=csv_bytes,
                        file_name="churn_batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Error parsing uploaded CSV: {e}")
        else:
            st.info("Upload a CSV file containing customer records. The file must include all 18 feature columns.")
            with st.expander("Required CSV column schema"):
                st.code(", ".join(required_features), language="text")

    except FileNotFoundError:
        st.error("Models not found. Run `python -m src.train` to train the models first.")
    except Exception as e:
        st.error(f"Error: {e}")


# ============================================================
# PAGE 5: High-Risk Customers
# ============================================================

elif page == "High-Risk Customers":
    st.markdown('<p class="page-title">High-risk customers</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Browse and export subscribers with elevated predicted churn probability.</p>', unsafe_allow_html=True)

    try:
        results_df, eval_results = get_dataset_predictions()
        best_name = eval_results['best_model_name']

        st.markdown('<div class="section-header"><h2>Filter and search</h2></div>', unsafe_allow_html=True)

        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            risk_filter = st.multiselect(
                "Risk level",
                options=["High Risk", "Medium Risk", "Low Risk"],
                default=["High Risk"]
            )
        with fc2:
            min_prob = st.slider("Minimum churn probability (%)", 0, 100, 70)
        with fc3:
            search_query = st.text_input("Search customer reference or details", "")

        filtered_df = results_df[
            (results_df['Risk Level'].isin(risk_filter)) &
            (results_df['Churn_Prob_Raw'] >= (min_prob / 100.0))
        ]

        if search_query:
            query_lower = search_query.lower()
            mask = filtered_df.astype(str).apply(lambda row: row.str.lower().str.contains(query_lower).any(), axis=1)
            filtered_df = filtered_df[mask]

        sort_col = st.selectbox(
            "Sort by",
            options=["Churn Probability (%)", "monthly_usage_rs", "Customer Reference"],
            index=0
        )
        filtered_df = filtered_df.sort_values(by=sort_col, ascending=False)

        display_cols = [
            'Customer Reference', 'Churn Prediction', 'Churn Probability (%)', 'Risk Level',
            'Network', 'service_type', 'relation_durattion', 'monthly_usage_rs',
            'contact_with_cust_care', 'Happy_with_intrnet_quality', 'recommend_to_others'
        ]
        other_cols = [c for c in filtered_df.columns if c not in display_cols and c != 'Churn_Prob_Raw']
        final_display_cols = display_cols + other_cols

        st.markdown('<div class="section-header"><h2>Customer records</h2></div>', unsafe_allow_html=True)
        st.markdown(f"**{len(filtered_df):,} record(s) matching current filters.**")

        if len(filtered_df) == 0:
            st.info("No customers match the selected filters. Adjust the risk level selection or minimum probability.")
        else:
            st.dataframe(
                filtered_df[final_display_cols],
                use_container_width=True,
                hide_index=True
            )

            csv_bytes = filtered_df[final_display_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export filtered list",
                data=csv_bytes,
                file_name="high_risk_customers.csv",
                mime="text/csv",
                use_container_width=True
            )

    except FileNotFoundError:
        st.error("Models not found. Run `python -m src.train` to train the models first.")
    except Exception as e:
        st.error(f"Error: {e}")


# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #AEB8CA; padding: 0.75rem; font-size: 0.8rem;">
    Customer Churn Prediction — Pakistani Telecom &nbsp;|&nbsp;
    Dataset: Pakistan Telecom (Kaggle) &nbsp;|&nbsp;
    Models: Logistic Regression, Random Forest, XGBoost &nbsp;|&nbsp;
    Explainability: SHAP
</div>
""", unsafe_allow_html=True)
