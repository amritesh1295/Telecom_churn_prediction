"""Shared cleaning, feature engineering and preprocessing.

This module is imported by BOTH the training notebook and the Flask app, so the
exact same transformations are applied at training time and at inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COL = "customerID"

# ---- Raw schema (IBM / Kaggle "Telco Customer Churn" dataset) --------------
RAW_CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
RAW_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# Allowed values per categorical field (used for form dropdowns + API validation)
YES_NO = ["Yes", "No"]
FIELD_CHOICES: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": YES_NO,
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": YES_NO,
    "OnlineBackup": YES_NO,
    "DeviceProtection": YES_NO,
    "TechSupport": YES_NO,
    "StreamingTV": YES_NO,
    "StreamingMovies": YES_NO,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}
# Values present in the raw dataset that we collapse to "No" during cleaning.
COLLAPSE_TO_NO = {"No internet service", "No phone service"}

# ---- Engineered features ----------------------------------------------------
ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
PROTECTION_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
ENGINEERED_NUMERIC = [
    "NumServices", "NumProtectionServices", "AvgMonthlySpend",
    "ChargeDelta", "AutoPay", "IsMonthToMonth",
]

NUMERIC_FEATURES = RAW_NUMERIC + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = RAW_CATEGORICAL + ["TenureGroup"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---- Steps ------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes, handle blanks and collapse redundant categories."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[ID_COL], errors="ignore")

    for col in RAW_CATEGORICAL:
        df[col] = df[col].str.strip()

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Brand-new customers (tenure == 0) have a blank TotalCharges in the raw file.
    df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])

    senior_map = {"yes": 1, "no": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0, "true": 1, "false": 0}
    df["SeniorCitizen"] = (
        df["SeniorCitizen"].astype(str).str.strip().str.lower().map(senior_map).fillna(0).astype(int)
    )

    for col in RAW_CATEGORICAL:
        df[col] = df[col].map(lambda v: "No" if v in COLLAPSE_TO_NO else v)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain features. Expects the output of :func:`clean`."""
    df = df.copy()
    yes = lambda col: (df[col] == "Yes").astype(int)  # noqa: E731

    df["NumServices"] = (
        sum(yes(c) for c in ADDON_SERVICES)
        + yes("PhoneService") + yes("MultipleLines")
        + (df["InternetService"] != "No").astype(int)
    )
    df["NumProtectionServices"] = sum(yes(c) for c in PROTECTION_SERVICES)

    safe_tenure = df["tenure"].clip(lower=1)
    df["AvgMonthlySpend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / safe_tenure, df["MonthlyCharges"])
    df["ChargeDelta"] = df["MonthlyCharges"] - df["AvgMonthlySpend"]  # >0 => bill rose vs. history

    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72, np.inf],
        labels=["0-12m", "13-24m", "25-48m", "49-72m", "73m+"],
    ).astype(str)
    df["AutoPay"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """clean -> engineer -> select model columns (target is not included)."""
    return engineer(clean(df))[MODEL_FEATURES]


def encode_target(y: pd.Series) -> pd.Series:
    return y.astype(str).str.strip().map({"Yes": 1, "No": 0}).astype(int)


def build_preprocessor() -> ColumnTransformer:
    """Scale numerics, one-hot encode categoricals. Output is a dense array (Keras-ready)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.0,
    )


def risk_level(probability: float) -> str:
    if probability >= 0.60:
        return "High"
    if probability >= 0.30:
        return "Medium"
    return "Low"
