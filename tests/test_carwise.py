from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.model_utils import load_artifacts
from train import clean_dataset

client = TestClient(app)


def test_clean_dataset_removes_exact_duplicates_and_invalid_rows() -> None:
    frame = pd.DataFrame([
        {"name": " Test Car ", "year": 2018, "selling_price": 400000, "km_driven": 25000, "fuel": "petrol", "seller_type": "individual", "transmission": "manual", "owner": "first owner"},
        {"name": " Test Car ", "year": 2018, "selling_price": 400000, "km_driven": 25000, "fuel": "petrol", "seller_type": "individual", "transmission": "manual", "owner": "first owner"},
        {"name": "Bad Car", "year": 1900, "selling_price": 400000, "km_driven": 25000, "fuel": "Petrol", "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner"},
    ])
    cleaned, evidence = clean_dataset(frame)
    assert len(cleaned) == 1
    assert evidence["duplicate_rows_removed"] == 1
    assert evidence["invalid_rows_removed"] == 1
    assert cleaned.iloc[0]["name"] == "Test Car"
    assert cleaned.iloc[0]["fuel"] == "Petrol"


def test_saved_artifacts_load() -> None:
    model, metadata, metrics = load_artifacts()
    assert model is not None
    assert metadata["features"]
    assert metrics["selected_model"]
    assert Path("artifacts/model.joblib").exists()


def test_health_endpoint_reports_loaded_model() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["status"] == "ok"


def test_metadata_contains_dataset_derived_options() -> None:
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset"]["rows_after_cleaning"] > 0
    assert body["options"]["name"]
    assert body["options"]["brand"]
    assert body["models_by_brand"][body["options"]["brand"][0]]
    assert len(body["model"]["comparisons"]) == 4


def first_valid_payload() -> dict:
    metadata = client.get("/metadata").json()
    return {
        "name": metadata["options"]["name"][0],
        "year": metadata["numeric_ranges"]["year"]["max"],
        "km_driven": 60000,
        "fuel": "Petrol" if "Petrol" in metadata["options"]["fuel"] else metadata["options"]["fuel"][0],
        "seller_type": "Individual" if "Individual" in metadata["options"]["seller_type"] else metadata["options"]["seller_type"][0],
        "transmission": "Manual" if "Manual" in metadata["options"]["transmission"] else metadata["options"]["transmission"][0],
        "owner": "First Owner" if "First Owner" in metadata["options"]["owner"] else metadata["options"]["owner"][0],
    }


def test_valid_prediction_returns_positive_price() -> None:
    response = client.post("/predict", json=first_valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["estimated_price"] > 0
    assert body["estimated_price_inr"].startswith("₹")
    assert body["model"]


def test_invalid_prediction_is_rejected() -> None:
    payload = first_valid_payload()
    payload["year"] = 1700
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_api_prediction_matches_direct_pipeline_prediction() -> None:
    payload = first_valid_payload()
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    model, _, _ = load_artifacts()
    direct = float(model.predict(pd.DataFrame([payload]))[0])
    assert response.json()["estimated_price"] == pytest.approx(round(max(0.0, direct), 2), abs=0.01)
