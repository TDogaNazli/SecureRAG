# main.py

import json
from evaluation.evaluate import run_full_evaluation

def load_patients(path: str):
    with open(path) as f:
        return json.load(f)

if __name__ == "__main__":
    print("🚀 Starting PrivMed-RAG Evaluation Pipeline...")

    # Path to your Synthea .parquet file
    synthea_path = "dataset/synthea-unified.parquet"

    run_full_evaluation(synthea_path)

    print("✅ Done.")
