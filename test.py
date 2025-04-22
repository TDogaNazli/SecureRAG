from retrieval.retrieve_subgraph import load_primekg
from evaluation.evaluate import evaluate_dataset, write_results
import pandas as pd
from tqdm import tqdm
from utils.load_from_parquet import load_patient_data

df = pd.read_parquet("dataset/synthea-unified.parquet")
patient_ids = df["patient_id"].unique()

for i, pid in enumerate(tqdm(patient_ids)):
    try:
        patient = load_patient_data(pid, "dataset/synthea-unified.parquet", privacy_level=0)
        if patient is None or len(patient) == 0:
            continue

        # Force risk question if no symptoms are available
        symptoms = patient.get("symptoms", [])
        print(symptoms)
    except Exception as e:
        print(f"❌ Error for patient {pid}: {e}")