# AI Financial Fraud Detection System — Version 2

An end-to-end machine learning system for detecting potentially fraudulent financial transactions and assigning them a **LOW, MEDIUM, or HIGH risk level**.

The project combines machine learning, a FastAPI backend, a SQLite transaction database, and a web-based monitoring dashboard to demonstrate how automated fraud screening can support preventive financial security.

---

## Project Overview

Financial fraud detection is challenging because fraudulent transactions are extremely rare compared with legitimate transactions.

This project uses a real-world public credit card transaction dataset containing **284,807 transactions**, including only **492 fraudulent transactions**. The system is designed to handle this severe class imbalance while providing an interpretable risk-based decision process.

The workflow is:

**Transaction Input → Data Processing → ML Prediction → Fraud Probability → Risk Classification → Preventive Action**

Depending on the predicted fraud probability, the system assigns:

| Risk Level | Preventive Action |
|------------|-------------------|
| LOW | ALLOW |
| MEDIUM | REVIEW |
| HIGH | BLOCK |

> **Note:** The ALLOW / REVIEW / BLOCK actions are simulated preventive actions for this academic/internship project. They do not actually interact with a bank, UPI network, card network, or financial account.

---

## Key Features

- Real public financial transaction dataset
- Handling of highly imbalanced fraud data
- Missing-value and duplicate checks
- Amount transformation using `log1p`
- Imbalance-aware Random Forest classifier
- Logistic Regression baseline
- Automatic decision-threshold selection
- Fraud probability prediction
- LOW / MEDIUM / HIGH risk classification
- ALLOW / REVIEW / BLOCK preventive actions
- FastAPI backend for real-time prediction
- SQLite database for transaction history
- Live transaction monitoring dashboard
- Model performance metrics
- Feature importance analysis
- Risk alerts and preventive explanations
- Simple user-friendly transaction input interface

---

## Dataset

The project uses the **Credit Card Fraud Detection** dataset released by ULB / European cardholders.

### Dataset Statistics

- **Total transactions:** 284,807
- **Fraudulent transactions:** 492
- **Legitimate transactions:** 284,315
- **Fraud rate:** approximately 0.172%
- **Features:** Time, V1–V28, Amount
- **Target:** Class

The V1–V28 variables are anonymized numerical features generated through PCA transformation. They represent patterns in the original transaction data rather than directly understandable fields such as location or transaction type.

### Dataset Source

The dataset can be obtained from:

- Zenodo: https://zenodo.org/records/7395559
- Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The raw CSV file is **not stored in this GitHub repository** because it is approximately 150 MB and exceeds GitHub's standard 100 MB file-size limit.

The project includes a script that can download the dataset during setup.

---

## System Architecture

```text
                   User
                    |
                    v
          Transaction Dashboard
                    |
                    v
             FastAPI Backend
                    |
                    v
          Input Validation
                    |
                    v
           Data Preprocessing
                    |
                    v
          Random Forest Model
                    |
                    v
          Fraud Probability
                    |
                    v
          Risk Decision Engine
             /      |       \
            /       |        \
          LOW     MEDIUM     HIGH
           |         |         |
        ALLOW     REVIEW     BLOCK
           \         |         /
            \        |        /
             v       v       v
             Transaction Log
                    |
                    v
              SQLite Database
                    |
                    v
             Monitoring Dashboard
