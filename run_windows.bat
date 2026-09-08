@echo off
setlocal
echo ================================================
echo AI FINANCIAL FRAUD DETECTION - VERSION 2
echo ================================================
if not exist venv python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist data\creditcard.csv python src\download_dataset.py
if not exist models\fraud_model.joblib python src\train_model.py
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
