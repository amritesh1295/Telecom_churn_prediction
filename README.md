# Telecom Customer Churn Prediction

End-to-end churn prediction system for a telecom operator:

* **Models:** Random Forest (scikit-learn) and Artificial Neural Network (TensorFlow/Keras)
* **Pipeline:** preprocessing, feature engineering, hyper-parameter tuning, model comparison, classification metrics
* **Serving:** Flask web form + REST API, with customers and predictions persisted in **MongoDB**

```
.
├── notebooks/churn_prediction.ipynb   # EDA -> features -> RF + ANN -> comparison -> export artifacts
├── src/features.py                    # cleaning + feature engineering + preprocessor (shared by notebook & API)
├── predictor.py                       # model loading, input validation, inference
├── db.py                              # MongoDB layer (customers + predictions collections)
├── app.py                             # Flask app (web UI + JSON API)
├── templates/index.html
├── tests/                             # offline unit tests (no Mongo / TensorFlow required)
├── data/                              # put the dataset CSV here (git-ignored)
├── models/                            # generated artifacts (git-ignored)
├── docker-compose.yml                 # local MongoDB
└── requirements.txt
```

## Quick start

```bash
# 1. environment (Python 3.10-3.12)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. dataset: download "Telco Customer Churn" (IBM sample data) from Kaggle and save as
#    data/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 3. train: run every cell of the notebook (writes models/*.joblib, models/ann_model.keras, models/metrics.json)
jupyter lab notebooks/churn_prediction.ipynb

# 4. MongoDB + API
docker compose up -d            # or point MONGO_URI at your own / Atlas cluster
cp .env.example .env
python app.py                   # http://localhost:5000
```

Production: `gunicorn -w 2 -b 0.0.0.0:5000 "app:create_app()"`

## Feature engineering

| Feature | Description |
|---|---|
| `NumServices`, `NumProtectionServices` | how many services / protection add-ons the customer has |
| `AvgMonthlySpend`, `ChargeDelta` | historical average bill and how far the current bill deviates from it |
| `TenureGroup` | tenure bucket |
| `AutoPay`, `IsMonthToMonth` | payment / contract behaviour flags |

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET`  | `/` | web form |
| `POST` | `/api/predict?model=all\|rf\|ann` | predict one customer (JSON) |
| `POST` | `/api/predict/batch` | `{"customers":[...]}`, max 500 |
| `GET`  | `/api/predictions?limit=20&risk_level=High&customer_id=...` | prediction history from MongoDB |
| `GET`  | `/api/customers/<customer_id>` | stored profile + last predictions |
| `GET`  | `/api/stats` | prediction counts by risk level |
| `GET`  | `/health` | model + MongoDB status |

```bash
curl -X POST http://localhost:5000/api/predict -H "Content-Type: application/json" -d '{
  "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 1, "PhoneService": "No", "MultipleLines": "No", "InternetService": "DSL",
  "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
  "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
}'
```

```json
{
  "customer_id": "7590-VHVEG",
  "churn_probability": 0.71,
  "churn_prediction": true,
  "risk_level": "High",
  "models": {
    "random_forest": {"probability": 0.69, "churn": true},
    "ann": {"probability": 0.73, "churn": true}
  },
  "stored": true,
  "prediction_id": "..."
}
```

`churn_probability` is the average of the available models. Risk bands: **High** >= 0.60, **Medium** >= 0.30, else **Low**.
If TensorFlow or `ann_model.keras` is absent the API serves the Random Forest alone; if MongoDB is down predictions still
work and the response contains `"stored": false`.

### MongoDB schema

* `customers`: `{customer_id (unique), profile{...}, created_at, updated_at}`
* `predictions`: `{customer_id, churn_probability, churn_prediction, risk_level, models, model_version, created_at}`

## Tests

```bash
python -m unittest discover -s tests -t . -v
```
The tests train a tiny model on synthetic data and use an in-memory fake of the MongoDB store, so they run anywhere.
Synthetic data is for tests only - model metrics must come from the real dataset in the notebook.

## Results

Fill in after running the notebook (`models/metrics.json` and the comparison table in section 7):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | | | | | |
| ANN | | | | | |

> Note: ~73% of customers in this dataset do not churn, so a "never churn" model already scores ~73% accuracy.
> Always read accuracy next to recall, F1 and ROC-AUC.
