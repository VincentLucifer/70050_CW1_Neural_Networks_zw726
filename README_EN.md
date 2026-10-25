# Neural Networks — IC Machine Learning Coursework

[简体中文](README.md) | [English](README_EN.md)

A neural networks coursework project for Machine Learning at Imperial College London (IC): a modular NumPy network with manual backpropagation, followed by a PyTorch regression pipeline for California Housing prices.

This implementation builds on university-provided starter code, interfaces and datasets. It demonstrates machine learning fundamentals and practical implementation; it does not claim that every component is independently authored. The original assignment description is preserved in the [coursework brief](docs/COURSEWORK_ORIGINAL.md).

## Core implementation

### Part 1: Neural networks in NumPy

- Forward and backward operations for `LinearLayer`, `SigmoidLayer` and `ReluLayer`, with cached intermediate values for gradient computation.
- Composable `MultiLayerNetwork` with `sigmoid`, `relu` and `identity` activations, alongside Xavier initialization, MSE / softmax cross-entropy losses and mini-batch SGD.
- Reversible min-max scaling in `Preprocessor`, including constant-feature handling.
- An Iris example using a 4 → 16 → 3 network, an 80/20 training/validation split, and printed loss and classification accuracy.

### Part 2: Housing regression in PyTorch

- A `Regressor` interface with `fit`, `predict` and `evaluate`; default hidden layers of 128 → 64 → 32, ReLU activations and a linear output.
- Float-column median imputation, min-max scaling, categorical encoding and restoration of the original target scale.
- Feature engineering with rooms per household, population per household, bedroom ratios, `log1p` features and Euclidean latitude/longitude distances to six California cities. These are coordinate-space approximations, not kilometres.
- Adam optimization, weight decay, cosine learning-rate scheduling and mini-batch training; evaluation uses RMSE on the original target scale.
- The entry point trains a validation model on an 80/20 split, then trains and saves a final model on all data, distinguishing validation error from final-model training error.

## Code navigation

| File | Responsibility |
| --- | --- |
| [part1_nn_lib.py](part1_nn_lib.py) | NumPy layers, losses, trainer, preprocessing and Iris example |
| [part2_house_value_regression.py](part2_house_value_regression.py) | PyTorch regressor, feature engineering, training and serialization |
| `iris.dat` / `housing.csv` | Iris / California Housing data supplied with the coursework repository |
| `part2_model.pickle` | Version-controlled model artifact; not loaded or evaluated for this update |
| [requirements.txt](requirements.txt) | Pinned Python dependencies |
| [docs/COURSEWORK_ORIGINAL.md](docs/COURSEWORK_ORIGINAL.md) | Original assignment description and submission instructions |

Stack: Python, NumPy for arrays and manual gradients, PyTorch for autograd and optimization, pandas for tabular processing, and scikit-learn for categorical encoding. See `requirements.txt` for all dependencies.

## Setup and usage

With Python and pip available, run from the repository root:

```powershell
python -m pip install -r requirements.txt

# Iris classification example: print loss / accuracy
python part1_nn_lib.py

# Evaluate a validation split, then train on all data and save
python part2_house_value_regression.py
```

The Part 1 example runs for 1,000 epochs. The Part 2 entry point trains two models for 200 epochs each. Running Part 2 overwrites `part2_model.pickle` in the working directory.

Commands are derived from source entry points and the dependency file. This documentation update did not install dependencies, train models or run tests; package availability and runtime compatibility remain unverified.

## Technical discussion and current limitations

Useful interview discussion points include the matrix gradients in `LinearLayer.backward`, reverse-order gradient propagation, reuse of training preprocessing parameters, and the distinction between validation and final submission models.

- Model metrics and LabTS results were not rechecked for this update; no unverified performance figures or coursework grades are claimed.
- Evaluation uses a single random holdout; no cross-validation, baseline comparison or separate experiment report is provided.
- Part 2 preprocessing targets the supplied housing data. Constant numeric columns, binary text categories and unseen categories need further handling.
- Part 2 sets seeds in `fit`, but the network is normally initialized in the constructor beforehand, so this does not ensure end-to-end reproducibility.
- Priorities for further work: control the full random state, record reproducible baseline and validation results, and improve preprocessing edge cases.

## Coursework provenance and original requirements

**Imperial College London — Machine Learning, Neural Networks coursework.** The university supplied the original task framework and data. This repository retains the coursework interfaces and brief and is not an official university release.

The original assignment requires LabTS testing, submission of the final commit, and inclusion of the final `part2_model.pickle` artifact. The GitHub copy supports personal organization and portfolio review; coursework submission follows the university workflow. The [preserved original brief](docs/COURSEWORK_ORIGINAL.md) includes the original environment and submission steps.
