# Customer Churn Prediction for Pakistani Telecom

A complete Machine Learning Web Application built with Python, pandas, scikit-learn, XGBoost, SHAP, and Streamlit to predict customer churn for Pakistani telecom networks (Jazz/Warid, Zong, Ufone, Telenor).

---

## 1. Project Overview & Objectives
Customer churn is a critical challenge in the telecommunications industry, where acquiring a new subscriber costs significantly more than retaining an existing one. This project provides:
- Machine learning models trained on Pakistani telecom customer data to classify churn risk.
- Interactive model performance comparisons across Logistic Regression, Random Forest, and XGBoost.
- Explainable AI using SHAP to highlight the exact features influencing each customer's prediction.
- A user-friendly Streamlit web interface for individual customer risk assessment and batch analytics.

---

## 2. Dataset Information & Source
- **Dataset**: Kaggle - [Pakistan Telecom Data Set](https://www.kaggle.com/datasets/zeshanali/pakistan-telecom-data-set) (`churnData.csv`)
- **Rows**: 9,760 records
- **Features**: 18 customer attributes (demographics, network operator, contract terms, package types, monthly usage, payment methods, complaints, customer service interactions).
- **Target**: `Churn` (Binary: `Yes` / `No`).
- **Operators Represented**: Jazz/Warid, Zong, Ufone, Telenor.

---

## 3. Technology Stack & Dependencies
- **Core Language**: Python 3.10+
- **Data Manipulation**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost
- **Explainable AI**: SHAP
- **Visualizations**: Plotly, Matplotlib, Seaborn
- **Web Application**: Streamlit

---

## 4. Installation & Environment Setup

### 1. Clone/Navigate to Repository
```bash
cd customer-churn-prediction
```

### 2. Create Virtual Environment & Install Requirements
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Model Training & Pipeline

To run the complete preprocessing, model training, evaluation, and artifact generation pipeline:

```bash
python3 -m src.train --data data/churnData.csv
```

This will output:
- Saved models: `models/all_models.pkl` and `models/best_model.pkl`
- Preprocessor: `models/preprocessor.pkl`
- Metrics: `models/evaluation_results.json`
- Test sets: `models/test_data.pkl`

---

## 6. Running the Streamlit Web Application

Launch the interactive web application:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your web browser.

---

## 7. Streamlit Dashboard Features

1. **📊 Executive Dashboard**: High-level KPI metrics, total customers, overall churn rate, model status, and distribution breakdown.
2. **🔍 Exploratory Data Analysis**: Data table preview, operator breakdown, monthly usage distributions, feature correlations.
3. **📈 Model Performance & Evaluation**: Side-by-side metric comparison (Accuracy, Precision, Recall, F1-Score, ROC-AUC), confusion matrices, and ROC curve overlay.
4. **🎯 Customer Churn Predictor**: Real-time prediction form for individual customers, displaying churn probability, risk level badge (Low/Medium/High), and SHAP feature contribution charts.

---

## 8. Google Colab Support
A Google Colab compatible notebook is located at `notebooks/exploration.ipynb`. It includes automated package installation cells for seamless execution in cloud notebook environments.

---

## 9. Project Directory Structure

```
customer-churn-prediction/
├── app.py                          # Streamlit Web Application
├── requirements.txt                # Dependency specifications
├── README.md                       # Project documentation
├── data/
│   └── churnData.csv               # Dataset file
├── models/                         # Saved artifacts & models
│   ├── best_model.pkl
│   ├── all_models.pkl
│   ├── preprocessor.pkl
│   ├── test_data.pkl
│   └── evaluation_results.json
├── src/
│   ├── data_loader.py              # Data ingestion and stats
│   ├── preprocessing.py            # Feature transformation pipeline
│   ├── train.py                    # Training orchestrator
│   ├── evaluate.py                 # Evaluation metrics & plots
│   └── explain.py                  # SHAP explainability
└── notebooks/
    └── exploration.ipynb           # EDA and experiment notebook
```

---

## 10. License & Acknowledgments
Dataset provided by Zeshan Ali on Kaggle under open access for educational and research purposes.
