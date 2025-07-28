# 🏦 Loan Approval Prediction - Kaggle Classification Challenge

This repository contains my solution for the [Loan Approval Predictions](https://www.kaggle.com/competitions/loan-approval-predictions) Kaggle competition.  
The primary objective of this challenge is to accurately predict the **probability of loan approval** (`loan_status`) based on various applicant and financial features.  
This is a **binary classification** task where the model is expected to estimate probabilities between 0 and 1.


---

## 🗂️ Project Structure

```

project-root/
├── 📄 task.ipynb # Main notebook: full ML pipeline from preprocessing to submission.
├── 📊 train.csv # Training dataset with features and target ('loan_status').
├── 🧪 test.csv # Test dataset: features for prediction.
├── 📝 sample_submission.csv # Template for Kaggle submission.
├── 🚀 sub.csv # Submission file from Logistic Regression model.
├── 🚀 sub1.csv # Submission file from XGBoost model.
└── 📜 README.md # Project documentation.

```
---

## 💻 Technologies Used

* **Programming Language**: Python 3.x
* **Notebook Environment**: Jupyter Notebook
* **Data Manipulation**: `pandas`, `numpy`
* **Visualization**: `seaborn`, `matplotlib` (used for exploratory plots like boxplots)
* **Machine Learning Frameworks**: `scikit-learn`, `xgboost`
    * `StandardScaler`: Feature normalization.
    * `LabelEncoder`: Encoding categorical variables.
    * `LogisticRegression`, `XGBClassifier`: Core ML models.
    * `Pipeline`: Seamless model-building workflow.
    * `GridSearchCV`: For exhaustive hyperparameter tuning.
    * `classification_report`: Model evaluation.
* **Other**: `warnings` (to suppress warnings during execution)

---

## 📊 Dataset Description

The dataset is provided by Kaggle under the competition: [Loan Approval Predictions](https://www.kaggle.com/competitions/loan-approval-predictions).

- `train.csv`: Historical data used to train the model, includes a binary target column `loan_status`.
- `test.csv`: Unlabeled data to be predicted and submitted.
- `sample_submission.csv`: Template showing the format required for submissions.

---

## 🔁 Project Workflow

The `task.ipynb` notebook executes the following steps:

1. **Data Loading & Preprocessing**
    - Reads both `train.csv` and `test.csv`.
    - Drops the non-predictive `id` column.
    - Encodes categorical features using `LabelEncoder`.

2. **Exploratory Data Analysis (EDA)**
    - Visualizes distributions using boxplots (`loan_amnt`).
    - Identifies feature scales and data types.

3. **Feature Engineering & Splitting**
    - Separates features and labels from the training set.
    - Splits training data into 80/20 train-validation subsets.

4. **Model 1: Logistic Regression (with Pipeline)**
    - Pipeline includes `StandardScaler` and `LogisticRegression`.
    - `GridSearchCV` is used to optimize:
        - `penalty`: l1, l2, elasticnet, None
        - `solver`: liblinear, saga, lbfgs, etc.
        - `l1_ratio`, `max_iter`, etc.
    - Evaluates the model using `classification_report`.

5. **Model 2: XGBoost Classifier**
    - Applies `XGBClassifier` for tree-based learning.
    - Uses `GridSearchCV` to tune `n_estimators`.

6. **Evaluation & Metrics**
    - Validation results reported via accuracy, precision, recall, F1-score.
    - Logistic and XGBoost models compared based on validation performance.

7. **Prediction & Submission**
    - Both models predict class probabilities on test data.
    - Outputs saved as:
        - `sub.csv` → Logistic Regression
        - `sub1.csv` → XGBoost

---

## 📈 Results Summary

| Model               | Tuned With       | Validation Metric | Output File |
|--------------------|------------------|-------------------|-------------|
| Logistic Regression | Pipeline + GridSearchCV | `classification_report` (F1, Recall) | `sub.csv` |
| XGBoost Classifier  | GridSearchCV     | `classification_report` (F1, Recall) | `sub1.csv` |

*Note: Probabilities for class 1 (`loan approved`) are submitted.*

---

## ⚙️ Installation

To install all required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
