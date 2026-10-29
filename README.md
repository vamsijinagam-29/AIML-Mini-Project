# CarWise AI — Indian Used Car Price Predictor

CarWise AI is an end-to-end machine-learning web application for estimating used-car resale prices in Indian rupees. It uses the uploaded Car Dekho CSV as its only data source, trains real scikit-learn models, saves the selected pipeline with joblib, exposes FastAPI prediction endpoints, and renders a React dashboard on the same origin.

## Stack

The backend uses Python, pandas, NumPy, scikit-learn, joblib, FastAPI and Uvicorn. The frontend uses React, TypeScript, Vite and a custom CSS system focused on a dark automotive instrumentation aesthetic. The application has no database requirement: prediction history is intentionally stored in the browser's localStorage.

## Data and model workflow

The original file is preserved at `data/original.csv`. Running `python train.py` cleans only the eight columns that exist in the uploaded file: `name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, and `owner`. Text whitespace and known categorical spellings are normalized. Exact duplicates and malformed/unrealistic rows are removed only after being counted. Rows with missing required inputs are dropped rather than filled with invented vehicle specifications. Because the source does not contain a separate brand column, the app derives a transparent brand index from each existing vehicle name and returns a `models_by_brand` mapping; the UI uses this mapping to filter the model selector after a brand is chosen, without adding brand as a fabricated ML feature.

The script splits the cleaned data before fitting preprocessing. Numeric fields use median imputation and scaling; categorical fields use most-frequent imputation and one-hot encoding. The candidate estimators are Dummy Regressor, Ridge Regression, Random Forest Regressor and Gradient Boosting Regressor. Each is evaluated on a held-out test set with MAE, RMSE and R². The selected model is the lowest-RMSE candidate, with MAE and R² shown alongside it, and is also checked with 3-fold cross-validation. After model selection, the chosen pipeline is refit on all cleaned rows for production inference and saved at `artifacts/model.joblib`. Actual metrics and comparisons are in `artifacts/metrics.json`; dataset-derived options and cleaning evidence are in `artifacts/metadata.json` and `artifacts/cleaning.json`.

## Run locally

```bash
python3 -m pip install -r requirements.txt
npm --prefix frontend install
python train.py
npm --prefix frontend run build
uvicorn backend.main:app --host 0.0.0.0 --port 3000
```

Open `http://localhost:3000`. Leave `VITE_API_BASE_URL` empty when frontend and backend share the same origin. For a separately served frontend, copy `.env.example` to `frontend/.env` and set `VITE_API_BASE_URL` to the FastAPI origin before building.

## API

`GET /health` reports whether the saved model artifact loaded. `GET /metadata` returns the supported model names, categorical values, numeric ranges, dataset statistics, selected model evidence and candidate comparison metrics. `POST /predict` accepts `name`, `year`, `km_driven`, `fuel`, `seller_type`, `transmission` and `owner`, validates each against the trained dataset metadata, then calls the saved pipeline for inference. It returns the estimated price, formatted INR value, selected model and disclaimer.

## Verification

Run `pytest -q` after training and building. The tests cover cleaning behavior, artifact loading, health, metadata, valid prediction output, invalid input handling and equality between direct joblib inference and the API response. The frontend is typechecked and bundled by `npm --prefix frontend run build`.

## Deployment shape

`Dockerfile` provides a multi-stage build: Node compiles `frontend/dist`, then a Python image runs FastAPI on `PORT` (default 3000). The `/health` endpoint is safe to use as the container readiness path. The managed project is configured with a server capability; publication should be performed only after the trained artifact, frontend build and smoke tests pass.

## Limitations and future improvement

The CSV is a historical listing dataset, not a live marketplace feed, and it does not contain condition, city, insurance, accident history or engine specifications. The model cannot account for those omitted variables. Future work could add validated location and condition fields from a richer source, use time-aware evaluation, calibrate prediction intervals, monitor drift, and add a separate brand/model hierarchy only when the source data supports it.
