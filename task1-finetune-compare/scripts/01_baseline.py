"""Baseline: TF-IDF + Logistic Regression (category and priority)."""
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common import MODELS, RESULTS, SEED, load_splits, save_json, scores


def build():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=5.0, random_state=SEED)),
    ])


def run(target, train, val, test):
    model = build()
    t0 = time.perf_counter()
    model.fit(train["text"], train[target])
    train_time = time.perf_counter() - t0

    val_pred = model.predict(val["text"])
    test_pred = model.predict(test["text"])

    # inference time per ticket (one at a time, like the API does)
    sample = test["text"].tolist()[:500]
    model.predict_proba(sample[:10])  # warm-up
    t0 = time.perf_counter()
    for s in sample:
        model.predict_proba([s])
    infer_ms = (time.perf_counter() - t0) / len(sample) * 1000

    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / f"baseline_{target}.joblib")
    test.assign(pred=test_pred).to_csv(RESULTS / f"baseline_{target}_test_preds.csv", index=False)

    result = {
        "model": "Baseline (TF-IDF + LogReg)",
        "target": target,
        "val": scores(val[target], val_pred),
        "test": scores(test[target], test_pred),
        "train_time_s": round(train_time, 3),
        "inference_ms_per_ticket": round(infer_ms, 3),
    }
    save_json(result, RESULTS / f"baseline_{target}_metrics.json")
    print(f"[{target}] test acc={result['test']['accuracy']:.4f} "
          f"macroF1={result['test']['macro_f1']:.4f} "
          f"train={train_time:.2f}s infer={infer_ms:.2f}ms/ticket")


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    tr, va, te = load_splits()
    for tgt in ("category", "priority"):
        run(tgt, tr, va, te)
