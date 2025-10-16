# Credit Card Fraud Detection

## Overview
A machine learning project to detect fraudulent credit card transactions using Python, Pandas, Scikit-learn, NumPy, Matplotlib, and Seaborn.  
The project uses **Logistic Regression** and **Random Forest** to classify transactions as **Fraudulent (1)** or **Legitimate (0)**.  
It also provides **real-time prediction** for single transactions.

---

## Features

### Machine Learning Models
- Random Forest Classifier
- Logistic Regression
- K-Nearest Neighbors (optional for experimentation)

### Data Processing
- Feature selection and preprocessing
- Train-test split for evaluation

### Evaluation Metrics
- Accuracy score
- Confusion matrix
- Classification report
- ROC-AUC score

### User Input Prediction
- Allows manual input of transaction features
- Predicts fraud probability in real-time

### Visualizations
- Fraud vs Non-Fraud distribution
- Feature correlation heatmap
- Random Forest feature importance

---

## Dataset
- File: `cdd.csv` (CSV format)
- Target column: `Class`  
  - 0 → Legitimate  
  - 1 → Fraudulent
- Features: `Time`, `Amount`, `V1` - `V28`

---

## Tools and Technologies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Dependencies
```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib

```
---
# Prerequisites
- Python must be installed on the system.
- You should have the dataset (cdd.csv).
- A code editor (preferred: VS Code or Jupyter Notebook).

# Installation and Setup

1.Clone the repository:
```sh
git clone https://github.com/Credit_card_fraud_detection
```
2.Install dependencies

3.Run the script:
```sh
python fraud_detection.py
```
# Usage
- Upload the dataset (cdd.csv).
- The script will train multiple machine learning models and evaluate their performance.
- The user can input transaction details for real-time fraud detection.

# Output
![image]()

