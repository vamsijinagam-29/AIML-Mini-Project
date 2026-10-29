# CarWise AI outcome checklist

## Dataset evidence and reproducible training

- [x] Preserve the uploaded CSV as the original source and generate a documented cleaned dataset from it.
- [x] Inspect actual columns, dtypes, row count, missing values, exact duplicates and invalid ranges; document every cleaning decision and the before/after row counts.
- [x] Split data before fitting preprocessing; use one-hot encoding for categorical features and appropriate numeric handling without leakage.
- [x] Train and compare DummyRegressor, Ridge or Linear Regression, Random Forest Regressor and Gradient Boosting Regressor.
- [x] Save actual held-out MAE, RMSE and R², cross-validation evidence where appropriate, comparison results, selected model and serialized pipeline with joblib.

## Real FastAPI prediction service

- [x] Implement `GET /health` to confirm backend availability and model loading status.
- [x] Implement `GET /metadata` with dataset-derived supported vehicle names, categorical options, numeric ranges, dataset statistics and real model metrics.
- [x] Implement `POST /predict` with typed validation, realistic ranges, supported vehicle validation and actual inference through the saved trained pipeline.
- [x] Configure CORS, useful errors, safe malformed-input handling, and a model artifact loading path that is shared with direct Python inference.

## CarWise AI dashboard

- [x] Build a premium responsive dark automotive dashboard titled “CarWise AI — Indian Used Car Price Predictor”.
- [x] Include the vehicle detail form using only source-supported fields: model name, year, kilometres, fuel, seller type, transmission and ownership category.
- [x] Send the form to the real prediction API, display the returned INR estimate, show loading and error states, and do not silently mock predictions.
- [x] Show model information, supported data range, dataset statistics, real evaluation metrics, model comparison visualization, local browser prediction history and an estimate disclaimer.

## Verification and delivery evidence

- [x] Add automated tests for cleaning/preprocessing, artifact loading, valid and invalid prediction requests, health, direct-vs-API prediction consistency and response validity.
- [x] Add frontend dependency manifest, `.env.example`, README setup instructions, architecture explanation, workflow diagram, evaluation report and future improvement suggestions.
- [x] Run tests, frontend build, API smoke checks and route-manifest verification; fix actionable errors before delivery.
- [x] Only claim deployment or end-to-end success when the running Preview service has been tested with evidence.
