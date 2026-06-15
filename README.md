# Project 2: Supervised Learning (Fraud Detection Pipeline)

This repository contains an end-to-end classification pipeline built to identify fraudulent financial transactions within highly imbalanced environments. Developed during my industrial training at DecodeLabs, this project focuses on algorithmic precision and advanced resamples techniques.

## Architectural Engineering

1. **Class Imbalance Mitigation (SMOTE)**
   * Addressed severe minority distribution drops by implementing Synthetic Minority Over-sampling Technique (SMOTE). This synthesizes new logical variations within sparse regions instead of duplicating raw data.

2. **Data Leakage Defensive Guard**
   * Splitting logic was structurally locked: `train_test_split(stratify=y)` was executed **prior** to any feature scaling or SMOTE interpolation to eliminate empirical pollution of the validation set.

3. **Model Benchmarking**
   * Trained and optimized multiple linear and non-linear estimators (`Logistic Regression` and `Random Forest Classifier`) utilizing `Scikit-Learn`.

4. **Production-Ready Evaluation Metric**
   * Discarded baseline 'Accuracy' metrics due to distribution skewness. Models were strictly benchmarked based on **Precision**, **Recall (Sensitivity)**, and **ROC-AUC** scores to minimize false declines while catching maximum fraudulent events.

## Tech Stack
* **Libraries:** Scikit-Learn, Imbalanced-Learn, Pandas, NumPy

## How to Execute
1. Set up dependencies:
   ```bash
   pip install scikit-learn imbalanced-learn pandas numpy
