# config.py

# === PrimeKG Paths ===
PRIMEKG_NODES_PATH = "dataset/primekg/raw/nodes.tsv"
PRIMEKG_EDGES_PATH = "dataset/primekg/raw/edges.csv"

# === Subgraph Settings ===
K_HOP = 1  # You can change this later to 2 or more if needed

# === Anonymization Settings ===
PRIVACY_LEVEL = 2
K_ANONYMITY = 5
L_DIVERSITY = 2

QUESTION_TYPES = ["risk", "symptom"]
