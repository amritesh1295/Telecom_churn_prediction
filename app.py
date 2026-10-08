"""Flask app: web form + JSON API for telecom churn prediction, persisted in MongoDB."""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from db import MongoStore
from predictor import ChurnPredictor, ModelUnavailable, ValidationError, validate_customer
from src.features import FIELD_CHOICES

BASE_DIR = Path(__file__).resolve().parent
MAX_BATCH = 500
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("churn-app")


def create_app(predictor: ChurnPredictor | None = None, store: MongoStore | None = None) -> Flask:
    load_dotenv()
    app = Flask(__name__)

    if predictor is None:
        try:
            predictor = ChurnPredictor(os.getenv("MODEL_DIR", BASE_DIR / "models"))
        except ModelUnavailable as exc:
            log.error("%s", exc)
    store = store or MongoStore(os.getenv("MONGO_URI", "mongodb://localhost:27017"),
                                os.getenv("MONGO_DB", "telecom_churn"))

    # ------------------------------------------------------------------ helpers
    def require_model():
        if predictor is None:
            raise ModelUnavailable("Models are not trained yet. Run notebooks/churn_prediction.ipynb first.")
        return predictor

    def run_prediction(payload: dict, which: str = "all") -> dict:
        model = require_model()
        customer_id, record = validate_customer(payload)
        customer_id = customer_id or f"CUST-{uuid.uuid4().hex[:8].upper()}"
        result = model.predict([record], which)[0]
        prediction_id = store.save_prediction(customer_id, record, result, model.version)
        return {"customer_id": customer_id, "prediction_id": prediction_id,
                "stored": prediction_id is not None, **result}

    # ------------------------------------------------------------------- pages
    @app.get("/")
    def index():
        return render_template("index.html", choices=FIELD_CHOICES, form={}, errors={}, result=None,
                               models=predictor.available_models if predictor else [],
                               metrics=predictor.metrics if predictor else {},
                               recent=store.recent_predictions(10))

    @app.post("/predict")
    def predict_form():
        form = request.form.to_dict()
        result, errors, status = None, {}, 200
        try:
            result = run_prediction(form)
        except ValidationError as exc:
            errors, status = exc.errors, 400
        except ModelUnavailable as exc:
            errors, status = {"_": str(exc)}, 503
        return render_template("index.html", choices=FIELD_CHOICES, form=form, errors=errors, result=result,
                               models=predictor.available_models if predictor else [],
                               metrics=predictor.metrics if predictor else {},
                               recent=store.recent_predictions(10)), status

    # --------------------------------------------------------------------- API
    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify(error="Send a JSON body with Content-Type: application/json"), 400
        which = request.args.get("model", "all")
        try:
            return jsonify(run_prediction(payload, which))
        except ValidationError as exc:
            return jsonify(error="validation_failed", fields=exc.errors), 400
        except ModelUnavailable as exc:
            return jsonify(error=str(exc)), 503
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @app.post("/api/predict/batch")
    def api_predict_batch():
        payload = request.get_json(silent=True) or {}
        customers = payload.get("customers")
        if not isinstance(customers, list) or not customers:
            return jsonify(error='Body must be {"customers": [ {...}, ... ]}'), 400
        if len(customers) > MAX_BATCH:
            return jsonify(error=f"Max {MAX_BATCH} customers per request"), 400
        results, errors = [], []
        for i, item in enumerate(customers):
            try:
                results.append(run_prediction(item))
            except ValidationError as exc:
                errors.append({"index": i, "fields": exc.errors})
            except ModelUnavailable as exc:
                return jsonify(error=str(exc)), 503
        return jsonify(results=results, errors=errors), (200 if results else 400)

    @app.get("/api/predictions")
    def api_predictions():
        limit = request.args.get("limit", 20, type=int)
        return jsonify(store.recent_predictions(limit, request.args.get("customer_id"), request.args.get("risk_level")))

    @app.get("/api/customers/<customer_id>")
    def api_customer(customer_id):
        customer = store.get_customer(customer_id)
        return (jsonify(customer), 200) if customer else (jsonify(error="customer not found"), 404)

    @app.get("/api/stats")
    def api_stats():
        return jsonify(store.stats())

    @app.get("/health")
    def health():
        return jsonify(status="ok" if predictor else "degraded",
                       models=predictor.available_models if predictor else [],
                       mongodb="up" if store.ping() else "down")

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(error=exc.name.lower()), exc.code

    @app.errorhandler(Exception)
    def unhandled(exc):
        log.exception("Unhandled error")
        return jsonify(error="internal server error"), 500

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=os.getenv("FLASK_DEBUG") == "1")
