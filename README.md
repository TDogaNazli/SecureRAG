# 🩺 SecureMed-RAG

**SecureMed-RAG** is a privacy-aware clinical question answering pipeline that uses **synthetic EHR data** and **PrimeKG** to evaluate the effectiveness of Retrieval-Augmented Generation (RAG) under varying levels of patient data anonymization.

The system generates patient-specific questions, retrieves relevant subgraphs from PrimeKG, and compares responses from a base LLM and a RAG-enhanced model. It supports multiple privacy levels including k-anonymity and l-diversity.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/TDogaNazli/SecureRAG.git
cd SecureMed-RAG
```

### 2. Python Setup

This project uses **Python 3.8.10**. You can use `pyenv` or any other version manager to install it.

```bash
python3 --version  # should be 3.8.10
python3 -m venv .myenv
source .myenv/bin/activate
pip install -r requirements.txt
```

### 3. Get a Gemini API Key

- Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
- Generate your Gemini API key

Then create a `.env` file in the root directory:

```bash
echo "GEMINI_API_KEY=your-key-here" > .env
```

---

## 📦 Data Setup

### PrimeKG Download

```bash
mkdir -p dataset/primekg/raw
wget -O dataset/primekg/raw/edges.csv https://dataverse.harvard.edu/api/access/datafile/6180616
wget -O dataset/primekg/raw/nodes.tsv https://dataverse.harvard.edu/api/access/datafile/6180617
```

### Prepare Patient EHR Data (Synthetic)

```bash
mkdir -p dataset/synthea-dataset-100
wget -O dataset/synthea-dataset-100.zip https://github.com/lhs-open/synthetic-data/raw/main/record/synthea-dataset-100.zip
unzip dataset/synthea-dataset-100.zip -d dataset/synthea-dataset-100
python utils/preprocess_synthea_data.py
```

---

## ▶️ Run Evaluation

To run the evaluation pipeline:

```bash
python main.py
```

### 🔧 Optional Configuration

- If you're hitting **Gemini API quota**, edit this line in `answer_generator.py`:
    ```python
  time.sleep(10)
  ```
- To change the number of patients processed:
  ```python
  res = evaluate_dataset(patient_ids[:50], synthea_path, privacy_level=level, G=G)
  ```
  Edit this line in `evaluate.py`.

---

## 📁 Project Structure

```
SecureMed-RAG/
├── main.py                       # Entry point to run full evaluation
├── requirements.txt             # Python dependencies
├── .env                         # Your Gemini API key (not committed to Git)
│
├── dataset/
│   ├── primekg/                 # Contains downloaded PrimeKG edges/nodes
│   └── synthea-dataset-100/     # Raw and preprocessed Synthea EHR data
│   └── synthea-unified.parquet  # Preprocessed parquet version
│
├── evaluation/
│   ├── evaluate.py              # Evaluation logic, scoring, and result generation
│   └── output/                  # Contains evaluation result files
│
├── generation/
│   └── answer_generator.py      # Calls Gemini to answer medical questions
│
├── preprocessing/
│   └── anonymize_ehr.py         # Implements k-anonymity, l-diversity, and PII removal
│
├── retrieval/
│   └── retrieve_subgraph.py     # Loads PrimeKG and extracts k-hop patient subgraphs
│
└── utils/
    ├── load_from_parquet.py     # Load and filter EHR data per patient
    └── preprocess_synthea_data.py # Groups and serializes Synthea data
```

---

## 📊 Evaluation Output

After running `main.py`, results are saved to:

```
evaluation/output/
├── level0_output.json           # Accuracy metrics (no anonymization)
├── level1_output.json           # Accuracy metrics (PII removed)
├── level2_output.json           # Accuracy metrics (k-anonymity + l-diversity)
└── question_results.json        # Per-question breakdown (LLM vs RAG)
```

### 📈 Metrics Reported

Each level includes:
- `LLM_only_accuracy`
- `RAG_accuracy`
- `improvement`: how much RAG improves/worsens accuracy
- `groundedness`: proportion of RAG answers grounded in PrimeKG
- Per-question-type breakdown

Each entry in `question_results.json` looks like:

```json
{
  "question_id": 1,
  "patient_id": "a123...",
  "question_type": "RISK_ASSESSMENT",
  "question": "What risks should be considered...",
  "llm_answer": "Hypertension may increase risk of stroke.",
  "rag_answer": "Patients with hypertension taking ibuprofen have higher risk of stroke.",
  "base_correct": true,
  "rag_correct": true
}
```

---
