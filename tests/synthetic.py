"""Synthetic data in the Telco schema - FOR TESTS ONLY (never use it to report model performance)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import FIELD_CHOICES


def make_telco_like(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"customerID": [f"{i:04d}-TEST" for i in range(n)]})
    for field, options in FIELD_CHOICES.items():
        df[field] = rng.choice(options, n)
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 73, n)
    df["MonthlyCharges"] = np.round(rng.uniform(18, 118, n), 2)

    # reproduce raw-file quirks: redundant categories + blank TotalCharges for tenure == 0
    df.loc[df["InternetService"] == "No", ["OnlineSecurity", "TechSupport"]] = "No internet service"
    df.loc[df["PhoneService"] == "No", "MultipleLines"] = "No phone service"
    total = (df["tenure"] * df["MonthlyCharges"]).round(2).astype(str)
    total[df["tenure"] == 0] = " "
    df["TotalCharges"] = total

    logit = (-1.0 + 1.4 * (df["Contract"] == "Month-to-month") - 0.04 * df["tenure"]
             + 0.02 * (df["MonthlyCharges"] - 65) + 0.6 * (df["InternetService"] == "Fiber optic")
             + 0.4 * (df["PaymentMethod"] == "Electronic check"))
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return df
