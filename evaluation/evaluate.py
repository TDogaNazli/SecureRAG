import json
import os
from tqdm import tqdm
from typing import List, Dict
from datetime import datetime
import pandas as pd
import networkx as nx
import time
from collections import Counter

from preprocessing.anonymize_ehr import anonymize_data
from retrieval.retrieve_subgraph import get_subgraph_for_patient, load_primekg
from generation.question_generator import generate_question
from generation.answer_generator import generate_answer
from config import PRIMEKG_NODES_PATH, PRIMEKG_EDGES_PATH
from utils.load_from_parquet import load_patient_data

def is_answer_supported(answer: str, subgraph) -> bool:
    terms = set()
    for u, v, d in subgraph.edges(data=True):
        terms.add(str(u).lower())
        terms.add(str(v).lower())
        if 'relation' in d:
            terms.add(str(d['relation']).lower())

    answer_lower = str(answer).lower()

    matched_terms = [term for term in terms if term in answer_lower]
    print(f"✅ Found {len(matched_terms)} matching terms in answer.")
    return len(matched_terms) > 0


def calculate_metrics(results: List[Dict], privacy_level: int) -> Dict:
    level_results = [r for r in results if r["privacy_level"] == privacy_level]

    if not level_results:
        print(f"⚠️ No results found for privacy level {privacy_level} with 'question_type'.")
        return {}

    if "question_type" not in pd.DataFrame(level_results).columns:
        print("❌ 'question_type' not found in results.")
        return {}
    
    grouped = pd.DataFrame(level_results).groupby("question_type")
    overall = {"LLM_only_accuracy": 0, "RAG_accuracy": 0, "improvement": 0, "groundedness": 0}
    by_question_type = {}

    for qtype, group in grouped:
        llm_correct = group[group["method"] == "LLM-only"]["correct"].mean()
        rag_correct = group[group["method"] == "LLM+RAG"]["correct"].mean()
        improvement = rag_correct - llm_correct
        groundedness = group[group["method"] == "LLM+RAG"]["correct"].mean()

        by_question_type[qtype] = {
            "LLM_only_accuracy": llm_correct,
            "RAG_accuracy": rag_correct,
            "improvement": improvement,
            "groundedness": groundedness,
            "num_questions": len(group)
        }

    overall["LLM_only_accuracy"] = pd.DataFrame(level_results)[pd.DataFrame(level_results)["method"] == "LLM-only"]["correct"].mean()
    overall["RAG_accuracy"] = pd.DataFrame(level_results)[pd.DataFrame(level_results)["method"] == "LLM+RAG"]["correct"].mean()
    overall["improvement"] = overall["RAG_accuracy"] - overall["LLM_only_accuracy"]

    return {"overall": overall, "by_question_type": by_question_type}

import json
import os
import pandas as pd

def write_results(all_results: List[Dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Save overall metrics for each privacy level
    for level in [0, 1, 2]:
        metrics = calculate_metrics(all_results, privacy_level=level)
        metrics_path = os.path.join(out_dir, f"level{level}_output.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"\n✅ Saved metrics for Privacy Level {level} to: {metrics_path}")

    # Save detailed per-question results (LLM + RAG answers side by side)
    flat_results = []
    for result in all_results:
        # Only save one row per question (skip duplicates for each method)
        if result["method"] == "LLM-only":
            qid = result["question_id"]
            patient_id = result["patient_id"]
            question_type = result["question_type"]
            question = result["question"]
            base_correct = result["base_correct"] if "base_correct" in result else result["correct"]

            # Try to find the corresponding RAG result
            matching_rag = next((r for r in all_results if r["question_id"] == qid and r["method"] == "LLM+RAG"), None)
            llm_answer = result.get("answer", None)
            rag_correct = matching_rag["correct"] if matching_rag else None
            rag_answer = matching_rag["answer"] if matching_rag else None

            flat_results.append({
                "question_id": qid,
                "patient_id": patient_id,
                "question_type": question_type.upper(),
                "question": question,
                "llm_answer": llm_answer,
                "rag_answer": rag_answer,
                "base_correct": base_correct,
                "rag_correct": rag_correct
            })

    # Write to JSON
    questions_path = os.path.join(out_dir, "question_results.json")
    with open(questions_path, "w") as f:
        json.dump(flat_results, f, indent=4)

    print(f"\n📝 Saved question-level results to: {questions_path}")


def evaluate_dataset(patient_ids: List[str], synthea_path: str, privacy_level: int, G) -> List[Dict]:
    results = []
    question_id = 0

    for i, pid in enumerate(tqdm(patient_ids, desc=f"Privacy Level {privacy_level}")):
        try:
            patient = load_patient_data(pid, synthea_path, privacy_level=privacy_level)
            if patient is None or len(patient) == 0:
                continue

            subgraph = get_subgraph_for_patient(patient, G)

            # Force risk question if no symptoms are available
            symptoms = patient.get("symptoms", [])
            force_risk = not symptoms or len(symptoms) == 0

            question, qtype = generate_question(patient, subgraph, force_risk=force_risk)

            # LLM-only baseline
            llm_only_answer = generate_answer(question, patient, subgraph=None)
            llm_only_correct = is_answer_supported(llm_only_answer, subgraph)

            # RAG answer
            rag_answer = generate_answer(question, patient, subgraph=subgraph)
            rag_correct = is_answer_supported(rag_answer, subgraph)

            results.append({
                "question_id": question_id,
                "patient_id": pid,
                "privacy_level": privacy_level,
                "question_type": qtype,
                "question": question,
                "answer": llm_only_answer,
                "correct": llm_only_correct,
                "method": "LLM-only"
            })
            results.append({
                "question_id": question_id,
                "patient_id": pid,
                "privacy_level": privacy_level,
                "question_type": qtype,
                "question": question,
                "answer": rag_answer,
                "correct": rag_correct,
                "method": "LLM+RAG"
            })
            question_id += 1

        except Exception as e:
            print(f"❌ Error for patient {pid}: {e}")

    return results

def run_full_evaluation(synthea_path: str, out_dir="evaluation/output"):
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_parquet(synthea_path)
    patient_ids = df["patient_id"].unique()

    from retrieval.retrieve_subgraph import load_primekg
    from config import PRIMEKG_NODES_PATH, PRIMEKG_EDGES_PATH
    G = load_primekg(PRIMEKG_NODES_PATH, PRIMEKG_EDGES_PATH)

    all_results = []
    for level in [0, 1, 2]:
        res = evaluate_dataset(patient_ids[:50], synthea_path, privacy_level=level, G=G)
        all_results.extend(res)

    write_results(all_results, out_dir)
