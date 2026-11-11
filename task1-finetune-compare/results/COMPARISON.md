### Test-set comparison: category

| Model | Accuracy | Macro F1 | Training time | Inference time per ticket |
|---|---|---|---|---|
| Baseline (TF-IDF + LogReg) | 1.0000 | 1.0000 | 0.1 s | 1.01 ms |
| DistilBERT (fine-tuned) | 1.0000 | 1.0000 | 723.4 s | 24.39 ms |

#### Per-class F1 (category)

| Class | Baseline F1 | DistilBERT F1 | Winner |
|---|---|---|---|
| Account | 1.000 | 1.000 | Tie |
| Billing | 1.000 | 1.000 | Tie |
| General | 1.000 | 1.000 | Tie |
| Shipping | 1.000 | 1.000 | Tie |
| Technical | 1.000 | 1.000 | Tie |

- Only baseline wrong: 0 | Only DistilBERT wrong: 0 | Both wrong: 0

### Test-set comparison: priority

| Model | Accuracy | Macro F1 | Training time | Inference time per ticket |
|---|---|---|---|---|
| Baseline (TF-IDF + LogReg) | 1.0000 | 1.0000 | 0.1 s | 1.04 ms |

_DistilBERT results not found yet. Run `python scripts/02_finetune_distilbert.py --target priority`._
