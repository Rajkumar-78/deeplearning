# Task 1: Fine-tune DistilBERT and Compare

Compares a TF-IDF + Logistic Regression baseline with fine-tuned DistilBERT on the
same train/val/test splits.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd scripts
python 00_prepare_data.py                  # or: --csv your_part1_data.csv (text,category,priority)
python 01_baseline.py                      # baseline: accuracy, macro F1, train + inference time
python 02_finetune_distilbert.py --target category
python 02_finetune_distilbert.py --target priority    # optional
python 03_compare.py                       # writes results/COMPARISON.md
```

## Outputs
- `results/*_metrics.json`: accuracy, macro F1, training time, inference ms per ticket
- `results/*_test_preds.csv`: test predictions (used for per-class analysis)
- `results/COMPARISON.md`: comparison table + per-class F1 (where DistilBERT wins)
- `models/distilbert_category/` and `models/distilbert_priority/`: copy these into Task 2's `models/` folder

## Submission
1. Comparison table: `results/COMPARISON.md`
2. 5-line justification: fill in `JUSTIFICATION.md`

## Note
The bundled data is synthetic and templated, so the baseline scores ~100%.
Use your real Part 1 data to get meaningful numbers. DistilBERT results appear
only after you run step `02`. A GPU (e.g. Colab T4) makes training much faster.
