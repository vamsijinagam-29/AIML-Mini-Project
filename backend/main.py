from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
FRONTEND_DIST = ROOT / "frontend" / "dist"
PUBLIC_DIR = ROOT / "public"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing artifact: {path.name}. Run train.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


try:
    MODEL = joblib.load(MODEL_PATH)
    METADATA = load_json(METADATA_PATH)
    METRICS = load_json(METRICS_PATH)
    MODEL_LOAD_ERROR = None
except Exception as exc:  # surfaced through /health rather than crashing import diagnostics
    MODEL = None
    METADATA = {}
    METRICS = {}
    MODEL_LOAD_ERROR = str(exc)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=160)
    year: int = Field(ge=1980, le=2026)
    km_driven: int = Field(ge=0, le=1_000_000)
    fuel: str = Field(min_length=1, max_length=30)
    seller_type: str = Field(min_length=1, max_length=40)
    transmission: str = Field(min_length=1, max_length=30)
    owner: str = Field(min_length=1, max_length=50)

    @field_validator("fuel", "seller_type", "transmission", "owner")
    @classmethod
    def non_empty_category(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Category cannot be empty")
        return value.strip()


class PredictionResponse(BaseModel):
    estimated_price: float
    estimated_price_inr: str
    model: str
    input: PredictionRequest
    disclaimer: str


app = FastAPI(
    title="CarWise AI API",
    version="1.0.0",
    description="Actual scikit-learn inference API for Indian used-car price estimation.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def ensure_ready() -> None:
    if MODEL is None:
        raise HTTPException(status_code=503, detail=f"Model is not loaded: {MODEL_LOAD_ERROR}")


def format_inr(value: float) -> str:
    return f"₹{value:,.0f}"


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if MODEL is not None else "degraded",
        "model_loaded": MODEL is not None,
        "selected_model": METRICS.get("selected_model"),
        "artifact": MODEL_PATH.name,
        "error": MODEL_LOAD_ERROR,
    }


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    ensure_ready()
    return {
        "features": METADATA["features"],
        "target": METADATA["target"],
        "dataset": METADATA["dataset"],
        "numeric_ranges": METADATA["numeric_ranges"],
        "options": METADATA["options"],
        "models_by_brand": METADATA["models_by_brand"],
        "model": {
            "selected_model": METRICS["selected_model"],
            "selection_metric": METRICS["selection_metric"],
            "selection_rule": METRICS["selection_rule"],
            "test_rows": METRICS["test_rows"],
            "selected_metrics": METRICS["selected_metrics"],
            "cross_validation": METRICS["cross_validation"],
            "comparisons": METRICS["comparisons"],
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    ensure_ready()
    options = METADATA["options"]
    for field in ("name", "fuel", "seller_type", "transmission", "owner"):
        if getattr(payload, field) not in options[field]:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported {field.replace('_', ' ')}. Choose a value from the dataset-derived options.",
            )

    ranges = METADATA["numeric_ranges"]
    for field in ("year", "km_driven"):
        value = getattr(payload, field)
        if not ranges[field]["min"] <= value <= ranges[field]["max"]:
            raise HTTPException(
                status_code=422,
                detail=f"{field.replace('_', ' ').title()} must be between {ranges[field]['min']:,} and {ranges[field]['max']:,} for this dataset.",
            )

    frame = pd.DataFrame([payload.model_dump()])
    try:
        predicted = float(MODEL.predict(frame)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    predicted = max(0.0, predicted)
    return PredictionResponse(
        estimated_price=round(predicted, 2),
        estimated_price_inr=format_inr(predicted),
        model=METRICS["selected_model"],
        input=payload,
        disclaimer="This is a data-backed estimate, not a guaranteed sale price. Actual resale value depends on condition, location and market timing.",
    )


@app.get("/manus-routes.json", include_in_schema=False)
def route_manifest() -> JSONResponse:
    manifest = PUBLIC_DIR / "manus-routes.json"
    if manifest.exists():
        return JSONResponse(content=json.loads(manifest.read_text(encoding="utf-8")))
    return JSONResponse(content={"routes": [{"path": "/", "title": "CarWise AI"}]})


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/", include_in_schema=False)
    def frontend_not_built() -> dict[str, str]:
        return {"message": "CarWise AI API is running. Build the frontend with npm --prefix frontend run build."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "3000")), reload=False)
