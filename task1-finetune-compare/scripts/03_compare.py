"""Build the comparison table + per-class analysis from saved results."""
import json

import pandas as pd
from sklearn.metrics import classification_report

from common import RESULTS


def load(name, target):
    p = RESULTS / f"{name}_{target}_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt_row(r):
    return (f"| {r['model']} | {r['test']['accuracy']:.4f} | {r['test']['macro_f1']:.4f} | "
            f"{r['train_time_s']:.1f} s | {r['inference_ms_per_ticket']:.2f} ms |")


def main():
    lines = []
    for target in ("category", "priority"):
        rows = [r for r in (load("baseline", target), load("distilbert", target)) if r]
        if not rows:
            continue
        lines += [f"### Test-set comparison: {target}", "",
                  "| Model | Accuracy | Macro F1 | Training time | Inference time per ticket |",
                  "|---|---|---|---|---|"]
        lines += [fmt_row(r) for r in rows]
        lines.append("")

        try:
            b = pd.read_csv(RESULTS / f"baseline_{target}_test_preds.csv")
            d = pd.read_csv(RESULTS / f"distilbert_{target}_test_preds.csv")
        except FileNotFoundError:
            lines += ["_DistilBERT results not found yet. Run "
                      f"`python scripts/02_finetune_distilbert.py --target {target}`._", ""]
            continue
        rb = classification_report(b[target], b["pred"], output_dict=True, zero_division=0)
        rd = classification_report(d[target], d["pred"], output_dict=True, zero_division=0)
        lines += [f"#### Per-class F1 ({target})", "",
                  "| Class | Baseline F1 | DistilBERT F1 | Winner |", "|---|---|---|---|"]
        for cls in sorted(k for k in rb if k not in ("accuracy", "macro avg", "weighted avg")):
            fb, fd = rb[cls]["f1-score"], rd[cls]["f1-score"]
            win = "DistilBERT" if fd - fb > 0.01 else "Baseline" if fb - fd > 0.01 else "Tie"
            lines.append(f"| {cls} | {fb:.3f} | {fd:.3f} | {win} |")
        lines.append("")
        both_wrong = ((b["pred"] != b[target]) & (d["pred"] != d[target])).sum()
        only_b = ((b["pred"] != b[target]) & (d["pred"] == d[target])).sum()
        only_d = ((b["pred"] == b[target]) & (d["pred"] != d[target])).sum()
        lines += [f"- Only baseline wrong: {only_b} | Only DistilBERT wrong: {only_d} | Both wrong: {both_wrong}", ""]

    text = "\n".join(lines)
    (RESULTS / "COMPARISON.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
