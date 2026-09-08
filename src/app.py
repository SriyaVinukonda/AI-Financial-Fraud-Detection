from pathlib import Path
from datetime import datetime
import json
import sqlite3
import uuid

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parents[1]

bundle = joblib.load(BASE / "models" / "fraud_model.joblib")

model = bundle["model"]
FEATURES = bundle["features"]
THRESHOLD = bundle["threshold"]

DB = BASE / "data" / "transactions.db"


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AI Financial Fraud Detection",
    version="2.0"
)

app.mount(
    "/static",
    StaticFiles(directory=BASE / "static"),
    name="static"
)

templates = Jinja2Templates(
    directory=BASE / "templates"
)


# ============================================================
# DATABASE
# ============================================================

def db():
    con = sqlite3.connect(DB)

    con.row_factory = sqlite3.Row

    con.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            amount REAL,
            fraud_probability REAL,
            risk TEXT,
            action TEXT,
            status TEXT,
            alert TEXT,
            reason TEXT
        )
    """)

    con.commit()

    return con


# ============================================================
# USER-FRIENDLY TRANSACTION INPUT
# ============================================================

class Transaction(BaseModel):

    Amount: float = Field(ge=0)

    TransactionType: str

    Location: str

    Device: str

    Hour: int = Field(ge=0, le=23)


# ============================================================
# CONVERT USER INPUT → V1-V28
# ============================================================

def create_features(tx):

    """
    Converts simple human-readable transaction information
    into the feature format expected by the trained model.

    The original Credit Card Fraud dataset contains:
    Time, V1...V28, Amount.

    V1-V28 are PCA-transformed anonymized features, so a normal
    user cannot realistically type them.

    Therefore we create a representative feature vector from
    the understandable transaction characteristics.
    """

    amount = float(tx["Amount"])
    transaction_type = tx["TransactionType"]
    location = tx["Location"]
    device = tx["Device"]
    hour = int(tx["Hour"])


    # --------------------------------------------------------
    # Start with neutral values
    # --------------------------------------------------------

    v = {f"V{i}": 0.0 for i in range(1, 29)}


    # --------------------------------------------------------
    # Amount influence
    # --------------------------------------------------------

    amount_log = np.log1p(amount)

    v["V1"] = (amount_log - 5.0) / 3.0


    # --------------------------------------------------------
    # Transaction type
    # --------------------------------------------------------

    if transaction_type == "transfer":

        v["V2"] = -1.5
        v["V3"] = 1.2
        v["V7"] = -1.0

    elif transaction_type == "withdrawal":

        v["V2"] = -1.0
        v["V5"] = 0.8
        v["V9"] = -0.8

    else:
        # normal purchase
        v["V2"] = 0.1
        v["V3"] = 0.1


    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    if location == "unusual":

        v["V4"] = 2.5
        v["V10"] = -2.0
        v["V12"] = -1.5

    else:

        v["V4"] = 0.0
        v["V10"] = 0.0
        v["V12"] = 0.0


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if device == "new":

        v["V14"] = -2.5
        v["V17"] = -2.0
        v["V18"] = -1.2

    else:

        v["V14"] = 0.0
        v["V17"] = 0.0
        v["V18"] = 0.0


    # --------------------------------------------------------
    # Transaction hour
    # --------------------------------------------------------

    if hour >= 0 and hour <= 5:

        # Late-night transaction
        v["V20"] = 2.0
        v["V21"] = 1.5

    elif hour >= 6 and hour <= 22:

        # Normal hours
        v["V20"] = 0.0
        v["V21"] = 0.0

    else:

        v["V20"] = 1.0


    # --------------------------------------------------------
    # Combined suspicious behaviour
    # --------------------------------------------------------

    suspicious_score = 0

    if amount >= 1500:
        suspicious_score += 1

    if transaction_type in ["transfer", "withdrawal"]:
        suspicious_score += 1

    if location == "unusual":
        suspicious_score += 1

    if device == "new":
        suspicious_score += 1

    if hour <= 5:
        suspicious_score += 1


    # Stronger signals when multiple suspicious conditions occur

    if suspicious_score >= 3:

        v["V22"] = 2.5
        v["V23"] = -2.0
        v["V24"] = 1.5
        v["V25"] = -1.5
        v["V26"] = 1.2
        v["V27"] = 1.0
        v["V28"] = -1.0

    elif suspicious_score == 2:

        v["V22"] = 1.0
        v["V23"] = -0.8
        v["V24"] = 0.5

    else:

        v["V22"] = 0.0
        v["V23"] = 0.0
        v["V24"] = 0.0


    # --------------------------------------------------------
    # Create final model input
    # --------------------------------------------------------

    data = {
        "Time": float(hour * 3600),
        **v,
        "Amount": float(np.log1p(amount))
    }


    # Make sure columns are EXACTLY what the model expects

    row = pd.DataFrame(
        [data],
        columns=FEATURES
    )

    return row


# ============================================================
# RISK DECISION
# ============================================================

def decision(prob):

    if prob >= max(0.75, THRESHOLD + 0.20):

        return (
            "HIGH",
            "BLOCK",
            "Critical fraud probability detected.",
            "Transaction BLOCKED to prevent potential financial loss."
        )


    if prob >= max(0.35, THRESHOLD):

        return (
            "MEDIUM",
            "REVIEW",
            "Suspicious transaction detected.",
            "Transaction HELD for manual verification before settlement."
        )


    return (
        "LOW",
        "ALLOW",
        "No strong fraud signal detected.",
        "Transaction ALLOWED after automated risk screening."
    )


# ============================================================
# HOME
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "threshold": THRESHOLD
        }
    )


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(tx: Transaction):

    # Get human-readable transaction details
    amount = float(tx.Amount)
    transaction_type = tx.TransactionType
    location = tx.Location
    device = tx.Device
    hour = tx.Hour

    # ---------------------------------------------------
    # 1. Get ML prediction
    # ---------------------------------------------------
    #
    # Your Random Forest was trained on V1-V28 + Amount.
    # We create a neutral/default feature vector for V1-V28
    # because normal users should NOT have to enter those values.
    #

    d = {
        "Time": float(hour)
    }

    for i in range(1, 29):
        d[f"V{i}"] = 0.0

    d["Amount"] = float(np.log1p(amount))

    row = pd.DataFrame([d], columns=FEATURES)

    ml_probability = float(model.predict_proba(row)[0, 1])

    # ---------------------------------------------------
    # 2. Human-readable risk signals
    # ---------------------------------------------------

    risk_score = 0.0

    # Large transaction
    if amount >= 50000:
        risk_score += 0.25
    elif amount >= 10000:
        risk_score += 0.15
    elif amount >= 5000:
        risk_score += 0.08

    # Transaction type
    if transaction_type == "transfer":
        risk_score += 0.15
    elif transaction_type == "withdrawal":
        risk_score += 0.10

    # Location
    if location == "unusual":
        risk_score += 0.25

    # Device
    if device == "new":
        risk_score += 0.20

    # Unusual transaction hour
    if hour <= 5 or hour >= 23:
        risk_score += 0.20

    # ---------------------------------------------------
    # 3. Combine ML + human-readable signals
    # ---------------------------------------------------

    probability = max(ml_probability, risk_score)

    # Strong suspicious combination
    if (
        transaction_type == "transfer"
        and location == "unusual"
        and device == "new"
        and (hour <= 5 or hour >= 23)
        and amount >= 10000
    ):
        probability = max(probability, 0.90)

    # ---------------------------------------------------
    # 4. Decide LOW / MEDIUM / HIGH
    # ---------------------------------------------------

    risk, action, alert, reason = decision(probability)

    # ---------------------------------------------------
    # 5. Save transaction
    # ---------------------------------------------------

    txid = "TX-" + uuid.uuid4().hex[:10].upper()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    con = db()

    con.execute(
        "INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?)",
        (
            txid,
            ts,
            amount,
            probability,
            risk,
            action,
            "AUTOMATED",
            alert,
            reason
        )
    )

    con.commit()
    con.close()

    return {
        "transaction_id": txid,
        "timestamp": ts,
        "fraud_probability": round(probability, 6),
        "risk": risk,
        "preventive_action": action,
        "alert": alert,
        "preventive_reason": reason,
        "threshold": THRESHOLD
    }

# ============================================================
# TRANSACTIONS
# ============================================================

@app.get("/transactions")
def transactions(limit: int = 50):

    con = db()

    rows = [
        dict(x)
        for x in con.execute(
            """
            SELECT *
            FROM transactions
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()
    ]

    con.close()

    return rows


# ============================================================
# STATISTICS
# ============================================================

@app.get("/stats")
def stats():

    con = db()

    q = lambda sql: con.execute(sql).fetchone()[0]

    out = {
        "total": q(
            "SELECT COUNT(*) FROM transactions"
        ),

        "high": q(
            "SELECT COUNT(*) FROM transactions WHERE risk='HIGH'"
        ),

        "review": q(
            "SELECT COUNT(*) FROM transactions WHERE action='REVIEW'"
        ),

        "blocked": q(
            "SELECT COUNT(*) FROM transactions WHERE action='BLOCK'"
        ),

        "allowed": q(
            "SELECT COUNT(*) FROM transactions WHERE action='ALLOW'"
        )
    }

    con.close()

    return out


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    metrics = json.loads(
        (
            BASE / "reports" / "metrics.json"
        ).read_text()
    )

    top = pd.read_csv(
        BASE / "reports" / "feature_importance.csv"
    ).head(8).to_dict(
        orient="records"
    )

    return {
        "metrics": metrics,
        "top_features": top
    }