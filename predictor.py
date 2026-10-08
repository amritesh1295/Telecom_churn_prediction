"""Model loading, request validation and inference (framework-agnostic)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import (
    FIELD_CHOICES, COLLAPSE_TO_NO, RAW_CATEGORICAL, prepare_features, risk_level,
)

log = logging.getLogger(__name__)
THRESHOLD = 0.5


class ValidationError(ValueError):
    def __init__(self, errors: dict[str, str]):
        super().__init__("Invalid input")
        self.errors = errors


class ModelUnavailable(RuntimeError):
    pass


def _number(payload, field, lo, hi, *, required=True, integer=False):
    raw = payload.get(field)
    if raw in (None, ""):
        if required:
            raise ValueError("required")
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError("must be a number") from None
    if not np.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"must be between {lo} and {hi}")
    return int(value) if integer else value


def validate_customer(payload) -> tuple[str | None, dict]:
    """Validate + normalise one customer. Returns (customer_id, clean_record)."""
    if not isinstance(payload, dict):
        raise ValidationError({"_": "expected a JSON object"})

    errors: dict[str, str] = {}
    record: dict = {}

    for field in RAW_CATEGORICAL:
        raw = payload.get(field)
        if raw in (None, ""):
            errors[field] = "required"
            continue
        allowed = {v.lower(): v for v in FIELD_CHOICES[field]}
        allowed.update({v.lower(): "No" for v in COLLAPSE_TO_NO})
        canonical = allowed.get(str(raw).strip().lower())
        if canonical is None:
            errors[field] = f"must be one of {FIELD_CHOICES[field]}"
        else:
            record[field] = canonical

    for field, args, kwargs in [
        ("tenure", (0, 120), {"integer": True}),
        ("MonthlyCharges", (0, 1000), {}),
        ("TotalCharges", (0, 200_000), {"required": False}),
    ]:
        try:
            record[field] = _number(payload, field, *args, **kwargs)
        except ValueError as exc:
            errors[field] = str(exc)

    senior = str(payload.get("SeniorCitizen", 0)).strip().lower()
    if senior not in {"0", "1", "0.0", "1.0", "yes", "no", "true", "false"}:
        errors["SeniorCitizen"] = "must be 0/1 or Yes/No"
    record["SeniorCitizen"] = senior

    if errors:
        raise ValidationError(errors)

    if record["TotalCharges"] is None:  # estimate if the caller did not send it
        record["TotalCharges"] = round(record["tenure"] * record["MonthlyCharges"], 2)

    customer_id = payload.get("customerID") or payload.get("customer_id")
    return (str(customer_id).strip() if customer_id else None), record


class ChurnPredictor:
    """Loads the preprocessor + Random Forest (+ ANN if TensorFlow and the model file exist)."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        pre, rf = model_dir / "preprocessor.joblib", model_dir / "rf_model.joblib"
        if not (pre.exists() and rf.exists()):
            raise ModelUnavailable(f"Model artifacts not found in {model_dir}. Run the notebook first.")

        self.preprocessor = joblib.load(pre)
        self.rf = joblib.load(rf)
        self.ann = self._load_ann(model_dir / "ann_model.keras")

        metrics_path = model_dir / "metrics.json"
        self.metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        self.version = self.metrics.get("trained_at", "unknown")

    @staticmethod
    def _load_ann(path: Path):
        if not path.exists():
            return None
        try:
            from tensorflow import keras  # imported lazily: optional at serving time
            return keras.models.load_model(path)
        except Exception as exc:  # noqa: BLE001
            log.warning("ANN model not loaded (%s). Serving Random Forest only.", exc)
            return None

    @property
    def available_models(self) -> list[str]:
        return ["random_forest"] + (["ann"] if self.ann is not None else [])

    def predict(self, records: list[dict], which: str = "all") -> list[dict]:
        if which not in {"all", "rf", "ann"}:
            raise ValueError("model must be one of: all, rf, ann")
        if which == "ann" and self.ann is None:
            raise ModelUnavailable("ANN model is not loaded on this server")

        X = self.preprocessor.transform(prepare_features(pd.DataFrame(records)))
        probs: dict[str, np.ndarray] = {}
        if which in ("all", "rf"):
            probs["random_forest"] = self.rf.predict_proba(X)[:, 1]
        if which in ("all", "ann") and self.ann is not None:
            probs["ann"] = np.asarray(self.ann.predict(X, verbose=0)).ravel()

        combined = np.mean(list(probs.values()), axis=0)
        results = []
        for i, p in enumerate(combined):
            p = float(p)
            results.append({
                "churn_probability": round(p, 4),
                "churn_prediction": p >= THRESHOLD,
                "risk_level": risk_level(p),
                "models": {
                    name: {"probability": round(float(v[i]), 4), "churn": bool(v[i] >= THRESHOLD)}
                    for name, v in probs.items()
                },
            })
        return results
