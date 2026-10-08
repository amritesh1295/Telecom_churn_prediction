"""MongoDB persistence for customers and predictions.

Collections
-----------
customers    one document per customer_id (latest profile, upserted)
predictions  one document per prediction (append-only history)

If MongoDB is unreachable the app keeps serving predictions; storage calls
return None / empty results instead of raising.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

try:
    from bson import ObjectId
    from pymongo import MongoClient, DESCENDING
    from pymongo.errors import PyMongoError
except ImportError:  # pymongo not installed
    MongoClient, ObjectId, DESCENDING, PyMongoError = None, None, -1, Exception

log = logging.getLogger(__name__)


def _serialise(obj):
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if ObjectId is not None and isinstance(obj, ObjectId):
        return str(obj)
    return obj


class MongoStore:
    def __init__(self, uri: str, db_name: str = "telecom_churn", timeout_ms: int = 2000):
        self.db_name = db_name
        self._indexed = False
        self._client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms) if MongoClient else None
        if self._client is None:
            log.warning("pymongo is not installed - predictions will not be stored.")

    @property
    def db(self):
        return self._client[self.db_name]

    def ping(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.admin.command("ping")
            return True
        except PyMongoError:
            return False

    def _ensure_indexes(self) -> None:
        if self._indexed:
            return
        self.db.customers.create_index("customer_id", unique=True)
        self.db.predictions.create_index([("created_at", DESCENDING)])
        self.db.predictions.create_index("customer_id")
        self.db.predictions.create_index("risk_level")
        self._indexed = True

    # ---- writes -------------------------------------------------------------
    def save_prediction(self, customer_id: str, profile: dict, result: dict, model_version: str = "") -> str | None:
        if self._client is None:
            return None
        now = datetime.now(timezone.utc)
        try:
            self._ensure_indexes()
            self.db.customers.update_one(
                {"customer_id": customer_id},
                {"$set": {"profile": profile, "updated_at": now}, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
            doc = {
                "customer_id": customer_id,
                "churn_probability": result["churn_probability"],
                "churn_prediction": result["churn_prediction"],
                "risk_level": result["risk_level"],
                "models": result["models"],
                "model_version": model_version,
                "created_at": now,
            }
            return str(self.db.predictions.insert_one(doc).inserted_id)
        except PyMongoError as exc:
            log.error("MongoDB write failed: %s", exc)
            return None

    # ---- reads --------------------------------------------------------------
    def recent_predictions(self, limit: int = 20, customer_id: str | None = None, risk_level: str | None = None):
        if self._client is None:
            return []
        query = {}
        if customer_id:
            query["customer_id"] = customer_id
        if risk_level:
            query["risk_level"] = risk_level
        try:
            cursor = self.db.predictions.find(query).sort("created_at", DESCENDING).limit(max(1, min(limit, 200)))
            return [_serialise(d) for d in cursor]
        except PyMongoError as exc:
            log.error("MongoDB read failed: %s", exc)
            return []

    def get_customer(self, customer_id: str):
        if self._client is None:
            return None
        try:
            customer = self.db.customers.find_one({"customer_id": customer_id})
            if not customer:
                return None
            history = self.recent_predictions(limit=10, customer_id=customer_id)
            return {**_serialise(customer), "recent_predictions": history}
        except PyMongoError as exc:
            log.error("MongoDB read failed: %s", exc)
            return None

    def stats(self) -> dict:
        if self._client is None:
            return {"total_predictions": 0, "by_risk_level": {}}
        try:
            rows = self.db.predictions.aggregate([{"$group": {"_id": "$risk_level", "n": {"$sum": 1}}}])
            by_risk = {r["_id"]: r["n"] for r in rows}
            return {"total_predictions": sum(by_risk.values()), "by_risk_level": by_risk,
                    "total_customers": self.db.customers.count_documents({})}
        except PyMongoError as exc:
            log.error("MongoDB read failed: %s", exc)
            return {"total_predictions": 0, "by_risk_level": {}}
