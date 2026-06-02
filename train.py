"""
src/train.py
CryptoShield AI – Model training, evaluation, and persistence.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import os, pickle, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)
from typing import Dict, Any, Optional

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def _eval(model, X_test, y_test, X_test_raw=None) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
        "precision_curve": precision_curve.tolist(),
        "recall_curve": recall_curve.tolist(),
        "y_prob": y_prob.tolist(),
        "y_test": y_test.tolist() if hasattr(y_test, "tolist") else list(y_test),
        "y_pred": y_pred.tolist(),
    }

def train_random_forest(pipeline_data: Dict[str, Any], n_estimators: int = 200,
                         max_depth: Optional[int] = None, random_state: int = 42) -> Dict[str, Any]:
    X_tr = pipeline_data["X_train"]
    y_tr = pipeline_data["y_train"]
    X_te = pipeline_data["X_test"]
    y_te = pipeline_data["y_test"]
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight="balanced", random_state=random_state, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    elapsed = round(time.time() - t0, 2)
    metrics = _eval(model, X_te, y_te)
    metrics["training_time_s"] = elapsed
    metrics["model_name"] = "Random Forest"
    fi = pd.Series(model.feature_importances_, index=pipeline_data["feature_names"]).sort_values(ascending=False)
    metrics["feature_importance"] = fi.to_dict()
    path = os.path.join(MODELS_DIR, "random_forest.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    metrics["model_path"] = path
    metrics["model"] = model
    return metrics

def train_xgboost(pipeline_data: Dict[str, Any], n_estimators: int = 200,
                  max_depth: int = 6, learning_rate: float = 0.1, random_state: int = 42) -> Dict[str, Any]:
    if not HAS_XGB:
        return {"error": "XGBoost not installed. Run: pip install xgboost"}
    X_tr = pipeline_data["X_train"]
    y_tr = pipeline_data["y_train"]
    X_te = pipeline_data["X_test"]
    y_te = pipeline_data["y_test"]
    scale_pos = int((y_tr == 0).sum() / (y_tr == 1).sum())
    t0 = time.time()
    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, scale_pos_weight=scale_pos,
        random_state=random_state, eval_metric="logloss",
        use_label_encoder=False, n_jobs=-1
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    elapsed = round(time.time() - t0, 2)
    metrics = _eval(model, X_te, y_te)
    metrics["training_time_s"] = elapsed
    metrics["model_name"] = "XGBoost"
    fi = pd.Series(model.feature_importances_, index=pipeline_data["feature_names"]).sort_values(ascending=False)
    metrics["feature_importance"] = fi.to_dict()
    path = os.path.join(MODELS_DIR, "xgboost.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    metrics["model_path"] = path
    metrics["model"] = model
    return metrics

def compare_models(rf_metrics: Dict, xgb_metrics: Dict) -> pd.DataFrame:
    rows = []
    for m in [rf_metrics, xgb_metrics]:
        if "error" in m:
            continue
        rows.append({
            "Model": m["model_name"],
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1 Score": m["f1"],
            "ROC-AUC": m["roc_auc"],
            "Training Time (s)": m["training_time_s"],
        })
    return pd.DataFrame(rows)

def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
