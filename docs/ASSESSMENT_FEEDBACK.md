# Assessment feedback — COMP70050 Neural Networks

Source: instructor's seven-page e-marking feedback for submission `zw726` (commit `854b5`). This document is a summary, **not a replacement for the original instructor-annotated PDF**.

## Automated tests

**40/40 — all listed hidden tests passed.** This is the automated-test result, not necessarily the final overall coursework grade.

| Component | Result |
| --- | ---: |
| Part 1: preprocessing, linear layer, activations, network and gradient descent | 20/20 |
| Part 2: general regressor test | 6/6 |
| Part 2: training-data performance threshold | 6/6 |
| Part 2: test-data performance threshold | 8/8 |
| **Total** | **40/40** |

- Recorded training RMSE: **41,289.8164** (threshold 90,000).
- Recorded test RMSE: **52,471.8047** (threshold 90,000).

## Instructor comments (translated/paraphrased)

1. **Feature engineering:** Creative ratio, log1p and city-distance features, but no evidence that they improve performance. Compare validation RMSE with and without `_augment` (ablation).
2. **Random seeds:** The seed is set in `fit`, after weights were created in `__init__`. Seed before model initialization to improve reproducibility.
3. **Comments:** Many comments repeat the next code line and some were added to provided code. Explain *why*, not just *what*.
4. **RMSE shapes (positive):** Using `reshape(-1)` on predictions and targets prevents `(n,1)` versus `(n,)` broadcasting bugs.
5. **Serialization design:** Changing `trained_model.__class__.__module__` inside a save function has process-wide side effects. Prefer training in a small script that imports the module.
6. **Validation workflow (positive):** Evaluate a validation probe, then retrain the final model on all available data.
7. **Generalisation:** Training RMSE ~41.3k versus held-out RMSE ~52.5k suggests a substantial gap; extra features and 200 epochs may contribute to overfitting. Tune learning rate, architecture, weight decay and epochs on validation RMSE.
8. **Numerical stability (positive):** Sigmoid input clipping prevents overflow in `np.exp`.
9. **Error message:** The text `-vincent` was accidentally left in an exception message.
10. **Scaling (positive):** Per-feature min/max scaling includes a zero-range guard for constant features.

## Interpretation and next steps

The code passed all automated tests; the instructor nevertheless highlighted limitations in experimental evidence, hyperparameter tuning, reproducibility, and code maintainability. Recommended next steps: run an ablation study, compare validation curves and tuning settings, set seeds before model construction, and remove stray error-message text and unnecessary comments.

**Provenance:** This is an individual student coursework implementation derived in part from university-provided starter code and data. It is not an official university publication. Consult course rules before redistributing course-provided files or the instructor's complete feedback.