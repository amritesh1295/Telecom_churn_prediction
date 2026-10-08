"""Offline tests: trains a tiny Random Forest on synthetic data, no MongoDB / TensorFlow needed.

Run:  python -m unittest discover -s tests -v      (or: pytest)
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

from app import create_app
from predictor import ChurnPredictor, ValidationError, validate_customer
from src import features as ft
from tests.synthetic import make_telco_like

VALID = {
    "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No", "tenure": 3,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 89.9,
}


class FakeStore:
    """In-memory stand-in for MongoStore."""
    def __init__(self):
        self.saved = []

    def save_prediction(self, customer_id, profile, result, model_version=""):
        self.saved.append((customer_id, profile, result))
        return f"pred-{len(self.saved)}"

    def recent_predictions(self, limit=20, customer_id=None, risk_level=None):
        return [{"customer_id": c, **r, "created_at": "2026-01-01T00:00:00"} for c, _, r in self.saved][:limit]

    def get_customer(self, cid):
        return next(({"customer_id": c, "profile": p} for c, p, _ in self.saved if c == cid), None)

    def stats(self):
        return {"total_predictions": len(self.saved), "by_risk_level": {}}

    def ping(self):
        return True


class ChurnAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        model_dir = Path(cls.tmp.name)
        raw = make_telco_like(1500, seed=3)
        y = ft.encode_target(raw[ft.TARGET])
        X = ft.prepare_features(raw.drop(columns=[ft.TARGET]))
        pre = ft.build_preprocessor()
        rf = RandomForestClassifier(n_estimators=40, random_state=0).fit(pre.fit_transform(X), y)
        joblib.dump(pre, model_dir / "preprocessor.joblib")
        joblib.dump(rf, model_dir / "rf_model.joblib")
        cls.predictor = ChurnPredictor(model_dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.store = FakeStore()
        self.client = create_app(predictor=self.predictor, store=self.store).test_client()

    # -- validation ----------------------------------------------------------
    def test_validation_normalises_and_estimates_total_charges(self):
        cid, rec = validate_customer({**VALID, "Contract": "month-to-month", "customerID": " A1 "})
        self.assertEqual(cid, "A1")
        self.assertEqual(rec["Contract"], "Month-to-month")
        self.assertAlmostEqual(rec["TotalCharges"], 3 * 89.9, places=2)

    def test_validation_reports_all_bad_fields(self):
        with self.assertRaises(ValidationError) as ctx:
            validate_customer({**VALID, "Contract": "weekly", "tenure": -4, "gender": ""})
        self.assertEqual(set(ctx.exception.errors), {"Contract", "tenure", "gender"})

    # -- API -----------------------------------------------------------------
    def test_predict_returns_probability_and_stores(self):
        resp = self.client.post("/api/predict", json={**VALID, "customerID": "C-1"})
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertTrue(0 <= body["churn_probability"] <= 1)
        self.assertIn(body["risk_level"], {"Low", "Medium", "High"})
        self.assertEqual((body["customer_id"], body["stored"]), ("C-1", True))
        self.assertEqual(len(self.store.saved), 1)

    def test_predict_generates_customer_id_when_missing(self):
        body = self.client.post("/api/predict", json=VALID).get_json()
        self.assertTrue(body["customer_id"].startswith("CUST-"))

    def test_invalid_payload_is_400_and_not_stored(self):
        resp = self.client.post("/api/predict", json={**VALID, "InternetService": "Satellite"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("InternetService", resp.get_json()["fields"])
        self.assertEqual(self.store.saved, [])

    def test_missing_json_body_is_400(self):
        self.assertEqual(self.client.post("/api/predict", data="nope").status_code, 400)

    def test_ann_requested_but_unavailable_is_503(self):
        self.assertEqual(self.client.post("/api/predict?model=ann", json=VALID).status_code, 503)

    def test_batch_mixes_valid_and_invalid(self):
        resp = self.client.post("/api/predict/batch", json={"customers": [VALID, {**VALID, "tenure": "abc"}]})
        body = resp.get_json()
        self.assertEqual((resp.status_code, len(body["results"]), len(body["errors"])), (200, 1, 1))
        self.assertEqual(body["errors"][0]["index"], 1)

    def test_form_page_and_submit(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        form = {k: str(v) for k, v in VALID.items()}
        resp = self.client.post("/predict", data=form)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"probability of churn", resp.data)

    def test_health_and_unknown_route(self):
        self.assertEqual(self.client.get("/health").get_json()["mongodb"], "up")
        self.assertEqual(self.client.get("/nope").status_code, 404)
        self.assertEqual(self.client.get("/api/predict").status_code, 405)

    def test_app_without_models_degrades_gracefully(self):
        empty = tempfile.mkdtemp()
        import os
        os.environ["MODEL_DIR"] = empty
        try:
            client = create_app(store=FakeStore()).test_client()
        finally:
            del os.environ["MODEL_DIR"]
        self.assertEqual(client.post("/api/predict", json=VALID).status_code, 503)
        self.assertEqual(client.get("/health").get_json()["status"], "degraded")


if __name__ == "__main__":
    unittest.main()
