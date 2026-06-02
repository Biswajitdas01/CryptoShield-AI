"""
src/predict.py
CryptoShield AI – Risk scoring and scam classification.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

RISK_THRESHOLDS = {"Low": 0.40, "Medium": 0.70, "High": 1.01}

def compute_risk_score(prob: float) -> int:
    """Map model probability [0,1] to risk score [0,100]."""
    return int(round(prob * 100))

def classify_risk(score: int) -> str:
    if score < 40:
        return "🟢 Low Risk"
    elif score < 70:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"

def predict_dataframe(model, df_features: pd.DataFrame,
                      feature_names: list, scaler=None) -> pd.DataFrame:
    """
    Run prediction on a DataFrame and return augmented results.
    """
    X = df_features[feature_names].copy()
    if scaler is not None:
        X_sc = scaler.transform(X)
    else:
        X_sc = X.values
    probs = model.predict_proba(X_sc)[:, 1]
    preds = model.predict(X_sc)
    scores = [compute_risk_score(p) for p in probs]
    labels = [classify_risk(s) for s in scores]
    result = df_features.copy()
    result["fraud_probability"] = probs.round(4)
    result["risk_score"] = scores
    result["risk_label"] = labels
    result["predicted_fraud"] = preds
    return result

def get_top_risky(result_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return result_df.nlargest(n, "risk_score")

def risk_summary(result_df: pd.DataFrame) -> Dict[str, Any]:
    total = len(result_df)
    high   = (result_df["risk_score"] >= 70).sum()
    medium = ((result_df["risk_score"] >= 40) & (result_df["risk_score"] < 70)).sum()
    low    = (result_df["risk_score"] < 40).sum()
    fraud  = result_df["predicted_fraud"].sum()
    return {
        "total_transactions": total,
        "fraud_detected": int(fraud),
        "fraud_pct": round(fraud / total * 100, 2),
        "high_risk": int(high),
        "medium_risk": int(medium),
        "low_risk": int(low),
        "avg_risk_score": round(result_df["risk_score"].mean(), 1),
    }

def explain_transaction(row: pd.Series, feature_importance: Dict[str, float],
                         top_n: int = 5) -> Dict[str, Any]:
    """
    Return human-readable explanation for a single flagged transaction.
    """
    risk_factors = []
    fi_sorted = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in fi_sorted[:top_n]:
        if feat in row.index:
            val = row[feat]
            risk_factors.append({
                "feature": feat,
                "value": round(float(val), 4) if isinstance(val, (int, float, np.floating)) else str(val),
                "importance": round(imp, 4),
            })
    return {
        "risk_score": int(row.get("risk_score", 0)),
        "risk_label": str(row.get("risk_label", "Unknown")),
        "fraud_probability": round(float(row.get("fraud_probability", 0)), 4),
        "top_risk_factors": risk_factors,
    }
