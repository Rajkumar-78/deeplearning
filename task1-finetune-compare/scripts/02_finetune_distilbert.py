"""Fine-tune DistilBERT with the Trainer API.

Usage:
    python scripts/02_finetune_distilbert.py --target category
    python scripts/02_finetune_distilbert.py --target priority   # if time allows
"""
import argparse
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

from common import MODELS, RESULTS, SEED, load_splits, save_json, scores

MODEL_NAME = "distilbert-base-uncased"


class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["category", "priority"], default="category")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()

    set_seed(SEED)
    target = args.target
    train, val, test = load_splits()

    labels = sorted(train[target].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = {n: TicketDataset(d["text"], d[target].map(label2id), tok, args.max_length)
          for n, d in (("train", train), ("val", val), ("test", test))}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id)

    out_dir = MODELS / f"distilbert_{target}"
    targs = TrainingArguments(
        output_dir=str(MODELS / f"_ckpt_{target}"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"],
                      eval_dataset=ds["val"], compute_metrics=compute_metrics,
                      data_collator=DataCollatorWithPadding(tok))

    t0 = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - t0

    # Evaluate on the SAME val / test sets as the baseline
    val_logits = trainer.predict(ds["val"]).predictions
    test_logits = trainer.predict(ds["test"]).predictions
    val_pred = [id2label[i] for i in val_logits.argmax(-1)]
    test_pred = [id2label[i] for i in test_logits.argmax(-1)]

    # Inference time per ticket (batch size 1, like the API)
    model.eval()
    dev = next(model.parameters()).device
    sample = test["text"].tolist()[:200]

    def one(text):
        enc = tok(text, truncation=True, max_length=args.max_length, return_tensors="pt").to(dev)
        with torch.no_grad():
            return model(**enc).logits

    for s in sample[:10]:
        one(s)  # warm-up
    t0 = time.perf_counter()
    for s in sample:
        one(s)
    infer_ms = (time.perf_counter() - t0) / len(sample) * 1000

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    test.assign(pred=test_pred).to_csv(RESULTS / f"distilbert_{target}_test_preds.csv", index=False)

    result = {
        "model": "DistilBERT (fine-tuned)",
        "target": target,
        "device": str(dev),
        "val": scores(val[target], val_pred),
        "test": scores(test[target], test_pred),
        "train_time_s": round(train_time, 3),
        "inference_ms_per_ticket": round(infer_ms, 3),
    }
    save_json(result, RESULTS / f"distilbert_{target}_metrics.json")
    print(f"[{target}] test acc={result['test']['accuracy']:.4f} "
          f"macroF1={result['test']['macro_f1']:.4f} "
          f"train={train_time:.1f}s infer={infer_ms:.1f}ms/ticket on {dev}")
    print(f"Saved model to {out_dir}")


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    main()
