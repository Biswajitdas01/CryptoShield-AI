"""
src/preprocessing.py
CryptoShield AI – Data preprocessing and feature engineering pipeline.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any

NUMERIC_FEATURES = [
    "amount_usd","gas_fee_usd","transaction_speed_s","num_transactions_24h",
    "wallet_age_days","unique_receivers_30d","avg_transaction_amount","std_transaction_amount",
]
BINARY_FEATURES = [
    "night_transaction","weekend_transaction","cross_chain","mixing_service_flag",
    "rapid_movement_flag","blacklist_interaction","contract_interaction",
]
CATEGORICAL_FEATURES = ["token_type", "network"]
TARGET = "is_fraud"
DROP_COLS = ["transaction_id", "sender_wallet", "receiver_wallet", "timestamp"]

def load_and_validate(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report: Dict[str, Any] = {"original_shape": df.shape, "errors": [], "warnings": []}
    required = set(NUMERIC_FEATURES + BINARY_FEATURES + [TARGET])
    missing_cols = required - set(df.columns)
    if missing_cols:
        report["errors"].append(f"Missing required columns: {missing_cols}")
    n_dup = df.duplicated().sum()
    if n_dup:
        report["warnings"].append(f"{n_dup} duplicate rows removed.")
        df = df.drop_duplicates()
    mv = df.isnull().sum()
    mv = mv[mv > 0]
    if len(mv):
        report["warnings"].append(f"Missing values detected. Filled with median/mode.")
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(exclude=[np.number]).columns:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) else "Unknown")
    report["cleaned_shape"] = df.shape
    report["fraud_rate"] = round(df[TARGET].mean() * 100, 2) if TARGET in df.columns else None
    return df, report

def get_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    total = df.shape[0]
    mv = df.isnull().sum().reset_index()
    mv.columns = ["Column", "Missing Count"]
    mv["Missing %"] = (mv["Missing Count"] / total * 100).round(2)
    mv["Data Type"] = [str(df[c].dtype) for c in mv["Column"]]
    return mv.sort_values("Missing %", ascending=False)

def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = df[num_cols].describe().T
    stats["skewness"] = df[num_cols].skew()
    stats["kurtosis"] = df[num_cols].kurt()
    return stats.round(4)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["amount_usd", "avg_transaction_amount", "std_transaction_amount"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
    if "amount_usd" in df.columns and "gas_fee_usd" in df.columns:
        df["gas_ratio"] = df["gas_fee_usd"] / (df["amount_usd"] + 1e-9)
    if "num_transactions_24h" in df.columns and "wallet_age_days" in df.columns:
        df["tx_per_day"] = df["num_transactions_24h"] / (df["wallet_age_days"] + 1)
    if "unique_receivers_30d" in df.columns and "num_transactions_24h" in df.columns:
        df["receiver_diversity"] = df["unique_receivers_30d"] / (df["num_transactions_24h"] + 1)
    flag_cols = [c for c in BINARY_FEATURES if c in df.columns]
    df["risk_flag_sum"] = df[flag_cols].sum(axis=1)
    return df

def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    encoders: Dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders

def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = set(DROP_COLS + [TARGET])
    return [c for c in df.columns if c not in exclude]

def preprocess_pipeline(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    df, validation_report = load_and_validate(df)
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_existing)
    df = engineer_features(df)
    df, encoders = encode_categoricals(df)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_sc, "X_test_scaled": X_test_sc,
        "feature_names": feature_cols,
        "scaler": scaler, "encoders": encoders,
        "validation_report": validation_report,
        "processed_df": df,
    }
