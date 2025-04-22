# utils/load_from_parquet.py

import pandas as pd
from preprocessing.anonymize_ehr import anonymize_data
import numpy as np

def load_patient_data(patient_id: str, synthea_path: str, privacy_level: int = 0) -> dict:
    """
    Load and anonymize patient data for a given ID from a Synthea parquet file.
    """
    try:
        df = pd.read_parquet(synthea_path)
        df = anonymize_data(df, level=privacy_level)
        patient_df = df[df['patient_id'] == patient_id]

        if patient_df.empty:
            print(f"Warning: Patient ID {patient_id} not found")
            return {}

        patient_dict = patient_df.iloc[0].to_dict()

        observations = patient_dict.get("observations", [])

        if isinstance(observations, np.ndarray):
            observations = observations.tolist()

        if isinstance(observations, list) and all(isinstance(o, dict) for o in observations):
            patient_dict["symptoms"] = [
            f'{o["description"]}: {o["value"]}'
            for o in observations
            if "description" in o and "value" in o and o["value"] is not None
        ]
        else:
            patient_dict["symptoms"] = []

        return patient_dict
    except Exception as e:
        print(f"Error loading patient data: {e}")
        return {}
