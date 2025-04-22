# generation/question_generator.py

from typing import Any, Dict, List, Tuple
import random
import numpy as np

def generate_question(patient_data: Dict, subgraph: Any, force_risk: bool = False) -> Tuple[str, str]:
    """
    Generate a natural language question and its type.
    Supports:
      - Risk Assessment
      - Symptom Progression
    Includes a justification clause from the KG to guide RAG.
    """

    # Extract from EHR
    conditions = ensure_list(patient_data.get("conditions", []))
    medications = ensure_list(patient_data.get("medications", []))
    symptoms = ensure_list(patient_data.get("symptoms", [])) 

    print("\n🩺 Generating question...")

    # Extract known facts from subgraph
    graph_facts = list(subgraph.edges(data=True))

    question_type = "risk" if force_risk else random.choice(["risk", "symptom"])
    print(f"  - selected question type: {question_type}")

    justification = ""
    if graph_facts:
        sample_facts = []
        for u, v, d in graph_facts:
            relation = d.get("relation", "related to")
            sample_facts.append(f"{u} {relation} {v}")
        justification = " Based on the knowledge graph which includes facts such as: " + "; ".join(sample_facts[:3]) + "."

    if question_type == "risk" and conditions and medications:
        cond = random.choice(conditions)
        med = random.choice(medications)
        question = f"What risks should be considered for a patient with {cond} taking {med}?{justification}"
        print("✅ Generated RISK question.")
        return question, "risk_assessment"

    elif question_type == "symptom" and conditions and symptoms:
        cond = random.choice(conditions)
        symptom = random.choice(symptoms)
        question = f"How might the symptom {symptom} progress for a patient with {cond}?{justification}"
        print("✅ Generated SYMPTOM question.")
        return question, "symptom_progression"

    else:
        print("⚠️ Could not match question type with available data. Using fallback.")
        question = f"What should be considered in managing this patient's case?{justification}"
        return question, "general"
    
def ensure_list(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, str):
        return [val]
    return val if isinstance(val, list) else list(val)