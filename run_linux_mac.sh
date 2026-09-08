#!/bin/bash
set -e
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
[ -f data/creditcard.csv ] || python src/download_dataset.py
[ -f models/fraud_model.joblib ] || python src/train_model.py
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
