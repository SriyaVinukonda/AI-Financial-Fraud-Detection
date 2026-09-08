from pathlib import Path
import requests

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "creditcard.csv"
URL = "https://zenodo.org/records/7395559/files/creditcard.csv?download=1"

if OUT.exists() and OUT.stat().st_size > 10_000_000:
    print("Dataset already exists:", OUT)
    raise SystemExit

print("Downloading the public credit-card fraud dataset (~150 MB)...")
with requests.get(URL, stream=True, timeout=180,
                  headers={"User-Agent":"FraudDetectionCollegeProject/2.0"}) as r:
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    done = 0
    with open(OUT, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"{done/1024/1024:.1f}/{total/1024/1024:.1f} MB", end="\r")
print("\nDataset saved:", OUT)
