from pathlib import Path
import json, sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix

BASE = Path(__file__).resolve().parents[1]
DATA = BASE/"data"/"creditcard.csv"
MODEL_DIR = BASE/"models"
REPORT_DIR = BASE/"reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError("data/creditcard.csv not found. Run python src/download_dataset.py")

df = pd.read_csv(DATA).drop_duplicates().reset_index(drop=True)
df["Amount"] = np.log1p(df["Amount"])
features = [c for c in df.columns if c != "Class"]
X, y = df[features], df["Class"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.20, stratify=y, random_state=42
)

baseline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1200, class_weight="balanced", solver="liblinear"))
])
baseline.fit(X_train, y_train)
base_prob = baseline.predict_proba(X_test)[:,1]

rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=180, max_depth=18, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    ))
])
rf.fit(X_train, y_train)
prob = rf.predict_proba(X_test)[:,1]

p, r, thresholds = precision_recall_curve(y_test, prob)
f1 = (2*p*r) / np.maximum(p+r, 1e-12)
idx = int(np.nanargmax(f1[:-1]))
threshold = float(np.clip(thresholds[idx], .10, .90))
pred = (prob >= threshold).astype(int)

metrics = {
    "dataset_rows": int(len(df)),
    "fraud_count": int(y.sum()),
    "fraud_rate_percent": round(float(y.mean()*100), 4),
    "baseline_logistic_roc_auc": round(float(roc_auc_score(y_test, base_prob)), 5),
    "random_forest_roc_auc": round(float(roc_auc_score(y_test, prob)), 5),
    "random_forest_pr_auc": round(float(average_precision_score(y_test, prob)), 5),
    "threshold": round(threshold, 5),
    "precision": round(float(precision_score(y_test,pred,zero_division=0)),5),
    "recall": round(float(recall_score(y_test,pred,zero_division=0)),5),
    "f1": round(float(f1_score(y_test,pred,zero_division=0)),5),
    "confusion_matrix": confusion_matrix(y_test,pred).tolist()
}
(REPORT_DIR/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")

rf_model = rf.named_steps["model"]
importance = pd.DataFrame({"feature":features,"importance":rf_model.feature_importances_}).sort_values("importance",ascending=False)
importance.to_csv(REPORT_DIR/"feature_importance.csv",index=False)

joblib.dump({"model":rf,"features":features,"threshold":threshold}, MODEL_DIR/"fraud_model.joblib")

print(json.dumps(metrics, indent=2))
print("\nTop features:\n", importance.head(10).to_string(index=False))
print("\nSaved:", MODEL_DIR/"fraud_model.joblib")
