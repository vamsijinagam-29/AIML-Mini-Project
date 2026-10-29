# CarWise AI evaluation report

This report is generated from the actual supplied CSV and the saved artifacts produced by `python train.py`.

## Dataset audit

- Source rows: **4,340**.
- Cleaned rows: **3,577**.
- Exact duplicates removed: **763**.
- Invalid-range rows removed: **0**.
- Missing values before/after: **0 / 0** across the supported columns.
- Supported source features: `name, year, km_driven, fuel, seller_type, transmission, owner`.
- Derived brand groups for cascading UI selection: **29**.

The original CSV remains unchanged at `data/original.csv`; the reproducible cleaned file is `data/cleaned.csv`. Cleaning decisions are recorded in `artifacts/cleaning.json` and `artifacts/metadata.json`.

## Held-out model comparison

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---:|---:|---:|
| Random Forest Regressor | ₹174,376.78 | ₹404,336.26 | 0.4925 |
| Gradient Boosting Regressor | ₹165,709.69 | ₹421,514.71 | 0.4484 |
| Ridge Regression | ₹191,657.62 | ₹424,380.39 | 0.4409 |
| Dummy Regressor | ₹283,304.75 | ₹582,844.14 | -0.0546 |

**Selected model:** Random Forest Regressor. It was selected by the lowest held-out RMSE because RMSE penalizes large resale-price errors; MAE and R² remain visible for context. Selected test metrics are MAE **₹174,376.78**, RMSE **₹404,336.26**, and R² **0.4925** on 716 held-out rows.
3-fold cross-validation for the selected pipeline produced mean MAE **₹175,588.03** (std ₹5,655.35), mean RMSE **₹357,824.22**, and mean R² **0.4705**.

## API and frontend evidence

The FastAPI app loads `artifacts/model.joblib` and exposes `/health`, `/metadata` and `/predict`. The React form sends actual JSON requests to `/predict`; it has no hardcoded price formula or silent mock fallback. The selected Random Forest pipeline produces differentiated estimates across materially different model/year/odometer combinations. Prediction history is browser-local by design.

## Test evidence

`PYTHONPATH=. pytest -q` and `npm --prefix frontend run build` are the required verification commands. Live smoke checks cover health, metadata, valid prediction, invalid input, route manifest, homepage rendering, and brand-to-model compatibility.
