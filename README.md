<div align="center">

# 🛡 CryptoShield AI
### Cryptocurrency Scam Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006400?style=for-the-badge)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**A production-grade AI/ML system for detecting fraudulent cryptocurrency transactions using Random Forest and XGBoost classifiers, with an interactive dark-themed Streamlit dashboard and explainability via SHAP.**

[🚀 Live Demo](#) · [📄 Report Sample](#) · [📦 Dataset](#data)

</div>

---

## 📌 Project Overview

CryptoShield AI is an end-to-end machine learning pipeline for cryptocurrency fraud detection. It takes raw transaction data, engineers features, trains ensemble classifiers, scores each transaction with a **Risk Score (0–100)**, and surfaces actionable insights through an interactive dashboard.

This project demonstrates:
- Real-world **imbalanced classification** handling
- **Explainable AI** (SHAP values, feature importance)
- Production-quality **Streamlit** UI with dark cybersecurity aesthetics
- **PDF report generation** for regulatory/audit use
- Clean, modular Python architecture suitable for team collaboration

---

## ✨ Features

| Module | Capabilities |
|---|---|
| 📁 **Dataset Upload** | CSV upload, validation, preview, summary stats |
| 🔬 **Data Analysis** | Missing values, distributions, correlation matrix, behaviour analysis |
| 🤖 **ML Training** | Random Forest + XGBoost, configurable hyperparameters, model persistence |
| ⚠️ **Scam Detection** | Risk scoring 0–100, Low/Medium/High classification, transaction inspector |
| 📊 **Dashboard** | KPIs, trend charts, heatmaps, pie charts, scatter plots |
| 🧠 **Explainable AI** | SHAP summary, waterfall plots, feature importance comparison |
| 📄 **PDF Reports** | Professional A4 reports for audit/compliance with ReportLab |

---

## 🏗 Project Architecture

```
CryptoShield-AI/
│
├── app.py                    ← Main Streamlit application (entry point)
├── requirements.txt          ← Python dependencies
├── README.md                 ← Project documentation
│
├── data/
│   ├── generate_sample.py    ← Synthetic dataset generator
│   └── crypto_transactions.csv ← 4,800-row sample dataset
│
├── models/                   ← Saved trained model files (.pkl)
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── reports/                  ← Generated PDF reports
│
├── screenshots/              ← UI screenshots for README/portfolio
│
├── notebooks/                ← Jupyter notebooks for EDA (optional)
│
├── src/                      ← Core ML/processing modules
│   ├── __init__.py
│   ├── preprocessing.py      ← Feature engineering, encoding, split/scale
│   ├── train.py              ← Model training, evaluation, persistence
│   ├── predict.py            ← Risk scoring and classification
│   ├── explainability.py     ← SHAP + Plotly explainability charts
│   └── report_generator.py  ← ReportLab PDF generator
│
└── assets/                   ← Logos, icons, static files
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/CryptoShield-AI.git
cd CryptoShield-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate Sample Dataset

```bash
python data/generate_sample.py
```

### 5. Launch the Application

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📊 Dataset Schema

| Column | Type | Description |
|---|---|---|
| `transaction_id` | string | Unique transaction hash |
| `sender_wallet` | string | Sender wallet address |
| `receiver_wallet` | string | Receiver wallet address |
| `timestamp` | datetime | Transaction timestamp |
| `amount_usd` | float | Transaction value in USD |
| `gas_fee_usd` | float | Gas/network fee in USD |
| `transaction_speed_s` | int | Confirmation time in seconds |
| `num_transactions_24h` | int | Sender's 24h transaction count |
| `wallet_age_days` | int | Age of sender's wallet in days |
| `unique_receivers_30d` | int | Unique receivers in 30 days |
| `avg_transaction_amount` | float | Sender's historical avg amount |
| `std_transaction_amount` | float | Standard deviation of amounts |
| `night_transaction` | bool | Transaction between 22:00–06:00 |
| `weekend_transaction` | bool | Transaction on weekend |
| `cross_chain` | bool | Cross-chain transfer flag |
| `mixing_service_flag` | bool | Known mixing service interaction |
| `rapid_movement_flag` | bool | Rapid fund movement detected |
| `blacklist_interaction` | bool | Interaction with blacklisted wallet |
| `contract_interaction` | bool | Smart contract interaction |
| `token_type` | string | ETH, BTC, USDT, BNB, USDC |
| `network` | string | Ethereum, Binance, Polygon, etc. |
| `is_fraud` | int | Ground truth label (0=legit, 1=fraud) |

---

## 🤖 Machine Learning Pipeline

```
Raw CSV
  │
  ▼
Validation & Cleaning
  │   • Duplicate removal
  │   • Missing value imputation
  │
  ▼
Feature Engineering
  │   • Log transforms (amount, avg, std)
  │   • Ratio features (gas_ratio, tx_per_day)
  │   • Risk flag aggregation
  │
  ▼
Encoding & Scaling
  │   • LabelEncoder for categoricals
  │   • StandardScaler for numerics
  │
  ▼
Train / Test Split (80/20, stratified)
  │
  ▼
Model Training
  │   • Random Forest (class_weight="balanced")
  │   • XGBoost     (scale_pos_weight=auto)
  │
  ▼
Evaluation
  │   • Accuracy, Precision, Recall, F1, ROC-AUC
  │   • Confusion Matrix, ROC Curves
  │
  ▼
Risk Scoring
    • Fraud probability → Risk Score [0–100]
    • Low (<40) | Medium (40–69) | High (≥70)
```

---

## 📈 Model Performance (Sample Dataset)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | ~0.97 | ~0.95 | ~0.92 | ~0.93 | ~0.99 |
| XGBoost | ~0.97 | ~0.94 | ~0.93 | ~0.94 | ~0.99 |

> Results on the synthetic 4,800-row sample dataset. Real-world results will vary.

---

## 🧠 Explainability

CryptoShield AI surfaces model decisions through:

- **Feature Importance Charts** — Bar charts ranked by model importance score
- **SHAP Summary Plots** — Mean absolute SHAP values across all test transactions
- **SHAP Waterfall (per transaction)** — Which features pushed a specific prediction toward fraud or legitimate
- **Plain-language Risk Factor Summary** — Top contributing features for each flagged transaction

---

## 🖥 Deployment Guide

### Streamlit Cloud (Recommended)

1. Push your repository to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set **Main file path** to `app.py`
4. Click **Deploy** — your app is live!

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python data/generate_sample.py
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t cryptoshield-ai .
docker run -p 8501:8501 cryptoshield-ai
```

### Heroku / Railway / Render

Add a `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## 🔧 Configuration & Customization

| Parameter | Location | Default | Description |
|---|---|---|---|
| Test split size | `app.py` → Training Config | 20% | Train/test split ratio |
| N Estimators | `app.py` → Training Config | 200 | Trees in ensemble |
| Max Depth | `app.py` → Training Config | None | Max tree depth |
| Risk thresholds | `src/predict.py` | 40 / 70 | Low/Medium/High cutoffs |
| Report output path | `src/report_generator.py` | `reports/` | PDF output directory |

---

## 🧩 Extending the Project

- **Add new models**: Add a function in `src/train.py` following the `train_random_forest` pattern
- **Add features**: Extend `NUMERIC_FEATURES` / `BINARY_FEATURES` in `src/preprocessing.py`
- **Custom risk logic**: Modify `compute_risk_score()` and `classify_risk()` in `src/predict.py`
- **Add pages**: Add a new `elif` block in `app.py`

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

Built as an AI/ML research portfolio project demonstrating production-grade fraud detection.  
Suitable for: **AI/ML Engineer portfolios · GitHub showcases · Graduate school applications · LinkedIn publications**

---

<div align="center">
<sub>⭐ If this project helped you, please give it a star!</sub>
</div>
