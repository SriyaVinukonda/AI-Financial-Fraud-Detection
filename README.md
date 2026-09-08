# AI Financial Fraud Detection System — Version 2

College-level end-to-end fraud detection project using a real public benchmark dataset.

## Public dataset
Credit Card Fraud Detection — ULB / European cardholders.
- 284,807 transactions
- 492 frauds
- about 0.172% fraud rate
- anonymized PCA features V1-V28, Time, Amount and Class

The raw CSV is not bundled because it is about 150 MB. The first run automatically downloads it from:
https://zenodo.org/records/7395559

Alternative:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Advanced features
- Real public dataset
- duplicate/missing-value checks
- Amount log transform
- class-imbalance-aware Random Forest
- Logistic Regression baseline
- ROC-AUC, PR-AUC, precision, recall and F1
- automatic decision threshold
- feature importance report
- FastAPI real-time scoring
- SQLite transaction history
- live monitoring dashboard
- explicit ALLOW / REVIEW / BLOCK preventive actions
- risk alerts
- model performance panel

## Windows
Open PowerShell in the project folder and run:

.un_windows.bat

First run downloads the public dataset and trains the model, so allow several minutes.

Then open http://127.0.0.1:8000

If automatic download fails, manually download creditcard.csv from the Zenodo page above and put it in data\creditcard.csv, then rerun.

This is an academic/demo system. Preventive actions are simulated and should not be connected to real financial accounts without security, compliance, model-governance and human-review controls.
