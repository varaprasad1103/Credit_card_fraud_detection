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
## Usage
- Run notebook `Credit_Card_Fraud_Detection.ipynb` or script `fraud_detection.py`
- Manual prediction example:
```python
sample_transaction = {'Time': 10000, 'V1': -1.35, 'V2': -0.07, 'V3': 2.53, 'Amount': 149.62, ...}
prediction, probability = predict_fraud(sample_transaction, model='rf')
print(prediction, probability)
# Output


Example Evaluation Output (from training & testing):

----- Logistic Regression Evaluation -----
Accuracy: 0.925
Confusion Matrix:
 [[17  2]
 [ 1 20]]
Classification Report:
               precision    recall  f1-score   support
           0       0.94      0.89      0.92        19
           1       0.91      0.95      0.93        21
Accuracy: 0.93
ROC-AUC Score: 0.9148

----- Random Forest Evaluation -----
Accuracy: 0.975
Confusion Matrix:
 [[18  1]
 [ 0 21]]
Classification Report:
               precision    recall  f1-score   support
           0       1.00      0.95      0.97        19
           1       0.95      1.00      0.98        21
Accuracy: 0.97
ROC-AUC Score: 1.0


```

---
![image](https://github.com/varaprasad1103/Credit_card_fraud_detection/blob/main/Screenshot%202025-10-16%20210610.png?raw=true)
![image](https://github.com/varaprasad1103/Credit_card_fraud_detection/blob/main/Screenshot%202025-10-16%20210724.png?raw=true)
![image](https://github.com/varaprasad1103/Credit_card_fraud_detection/blob/main/Screenshot%202025-10-16%20210755.png?raw=true)
```python

Prediction Example:
Prediction: Legitimate
Probability of Fraud: 0.2200

Prediction: Legitimate
Probability of Fraud: 0.2200
cdd.zip
cdd.zip(application/x-zip-compressed) - 32176 bytes, last modified: 10/16/2025 - 100% done
Saving cdd.zip to cdd (2).zip
Prediction	Fraud Probability
0	            Fraud	0.89
1	            Fraud	0.90
2	            Fraud	0.87
3	            Legitimate	0.17
4	            Fraud	0.88

