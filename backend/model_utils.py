from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parents[1]


def load_artifacts() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    artifacts = ROOT / "artifacts"
    model = joblib.load(artifacts / "model.joblib")
    metadata = json.loads((artifacts / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
    return model, metadata, metrics
