# Fraud Strategy Project

Cost-sensitive fraud detection comparing **rules**, **machine learning (XGBoost)**, and a **hybrid** strategy.

## Project flow

```text
Part 1  EDA + rules + full-timeline transform
   |    handoff: src/transform.prepare_train_val_test()
   v
Part 2  Train models (existing XGBoost reused; train features unchanged)
   |    handoff: artifacts/models/
   v
Part 3  Optimize thresholds on validation at FP=$30
   |    handoff: artifacts/strategy/frozen_strategy.json
   v
Part 4  Evaluate frozen strategy on test
        + robustness under FP costs $10/$30/$50/$100 (no re-tune)
```

## Feature prep (important)

`prepare_train_val_test()`:

1. Concatenate train-file + test-file chronologically
2. Run past-looking `create_features` once
3. Split train-file rows into train (80%) / val (20%)
4. Test = test-file rows with prior history

No global averages. Scaler/encoder remain train-only in the model pipeline. **No model retrain** required.

## Method

1. **Optimize** at FP=$30 on validation
2. **Freeze** thresholds/rules
3. **Test** once
4. **Robustness:** same frozen predictions under `$10, $30, $50, $100`

### Frozen @$30

- XGBoost threshold: `0.082`
- Hybrid: high `0.092`, low `0.016`

## Robustness results

| FP cost | Hybrid | XGBoost only | Rules only | Hybrid savings vs XGB | Hybrid % reduction vs XGB | Winner |
|---:|---:|---:|---:|---:|---:|:---|
| $10 | $66,404.48 | $92,041.04 | $318,045.56 | $25,636.56 | 27.85% | Hybrid |
| $30 | $104,504.48 | $116,141.04 | $408,025.56 | $11,636.56 | 10.02% | Hybrid |
| $50 | $142,604.48 | $140,241.04 | $498,005.56 | $-2,363.44 | -1.69% | XGBoost only |
| $100 | $237,854.48 | $200,491.04 | $722,955.56 | $-37,363.44 | -18.64% | XGBoost only |

- **$30 baseline hybrid vs XGBoost:** 10.0%
- **Mean across grid:** 4.4%
- **Range:** -18.6% – 27.9%
- Hybrid wins: **2 / 4**

## Notebooks

| Notebook | Role |
|---|---|
| [`notebooks/01_eda_and_rules.ipynb`](notebooks/01_eda_and_rules.ipynb) | EDA + transform handoff |
| [`notebooks/02_train_models.ipynb`](notebooks/02_train_models.ipynb) | Model training |
| [`notebooks/03_optimize_thresholds.ipynb`](notebooks/03_optimize_thresholds.ipynb) | Validation optimization @$30 |
| [`notebooks/04_test_evaluation.ipynb`](notebooks/04_test_evaluation.ipynb) | Test + robustness |

## Shared code / artifacts

- `src/` — transform, rules, metrics, evaluate, paths
- `artifacts/models/` — trained models (local)
- `artifacts/strategy/frozen_strategy.json` — Part 3→4 package
- `data/` — CSVs (local only, gitignored)
