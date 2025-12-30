# Diabetes Prediction with MLflow

A machine learning project for diabetes prediction using ensemble methods and stacking, with full experiment tracking via MLflow.

## Overview

This project demonstrates a complete ML workflow for binary classification on imbalanced diabetes data. The pipeline includes data preprocessing, K-Fold cross-validation, model training, stacking ensemble, and experiment tracking.

## Data

The dataset combines two sources with binary diabetes labels. The data exhibits class imbalance between positive (diabetic) and negative (non-diabetic) cases, which is addressed using SMOTE oversampling on the training set.

## Workflow

1. Data preprocessing and concatenation of multiple data sources
2. Stratified K-Fold cross-validation (5 folds)
3. SMOTE applied on each fold's training set
4. Base learners training: XGBoost, LightGBM, CatBoost
5. Out-of-Fold (OOF) predictions collection
6. Best fold selection per model based on AUC
7. Stacking with Logistic Regression as meta-learner
8. Final model comparison and evaluation

## MLflow Integration

All experiments are tracked using MLflow:

- Experiment: `Diabetes Prediction K-Folds` for base learners
- Experiment: `Diabetes Prediction - Stacking` for meta-learner

Logged items per run:
- Parameters: fold number, model type, SMOTE strategy
- Metrics: AUC, accuracy, recall, precision, F1 score
- Artifacts: trained models, meta-features CSV

Best models from each base learner and the final stacking model are registered in MLflow Model Registry.

## Project Structure

```
Diabetes-mlflow/
├── data/
│   ├── db1.csv
│   └── db2.csv
├── notebooks/
│   └── main.ipynb
├── mlruns/
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/pjazzy314159/diabetes-mlflow
cd diabetes-mlflow
```

### 2. Create virtual environment


```bash
conda create -n diabetes-mlflow python=3.10
conda activate diabetes-mlflow
```


### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook notebooks/main.ipynb
```

### 5. View MLflow UI

After running the notebook, start MLflow UI:
```bash
mlflow ui
```

Open browser at `http://localhost:5000` to view experiments.

## Results

The stacking ensemble combines predictions from three gradient boosting models. Performance comparison uses OOF predictions to ensure unbiased evaluation. Final metrics and model coefficients are logged to MLflow for reproducibility.