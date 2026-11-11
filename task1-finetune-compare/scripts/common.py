"""Shared helpers: paths, split loading, metrics."""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SPLITS = DATA / "splits"
SEED = 42


def load_splits():
    """Return train/val/test DataFrames (same files for baseline AND DistilBERT)."""
    out = {}
    for name in ("train", "val", "test"):
        path = SPLITS / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python scripts/00_prepare_data.py"
            )
        out[name] = pd.read_csv(path)
    return out["train"], out["val"], out["test"]


def scores(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
