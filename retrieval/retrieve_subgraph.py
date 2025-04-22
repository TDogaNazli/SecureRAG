import networkx as nx
import pandas as pd
from typing import List, Set

from config import PRIMEKG_NODES_PATH, PRIMEKG_EDGES_PATH, K_HOP

def load_primekg(nodes_path: str, edges_path: str) -> nx.Graph:
    """Load PrimeKG as a NetworkX graph"""
    print(f"📂 Loading PrimeKG nodes from: {nodes_path}")
    print(f"📂 Loading PrimeKG edges from: {edges_path}")
    G = nx.Graph()
    
    try:
        # Load nodes and edges
        nodes = pd.read_csv(nodes_path, sep="\t")
        edges = pd.read_csv(edges_path)
        print(f"✅ Loaded {len(nodes)} nodes and {len(edges)} edges.")
    except Exception as e:
        print(f"❌ Failed to load PrimeKG data: {e}")
        raise

    try:
        # Add nodes
        for idx, row in nodes.iterrows():
            if idx % 10000 == 0:
                print(f"    ➕ Added {idx} nodes so far...")
            G.add_node(row['node_id'], 
                       node_type=row['node_type'], 
                       node_name=row['node_name'], 
                       node_source=row['node_source'])

        # Precompute node_id lookup array
        node_ids = nodes['node_id'].values

        # Add edges
        for idx, row in edges.iterrows():
            if idx % 100000 == 0:
                print(f"    🔗 Processed {idx} edges so far...")
            try:
                source = node_ids[row['x_index']]
                target = node_ids[row['y_index']]
                G.add_edge(source, target, relation=row['relation'], display_relation=row['display_relation'])
            except IndexError:
                print(f"⚠️ Skipping invalid edge at row {idx} with indices {row['x_index']}, {row['y_index']}")

        print(f"✅ Finished building graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    except Exception as e:
        print(f"❌ Error while building graph: {e}")
        raise

    return G

RELEVANT_DISPLAY_RELATIONS = {
    "associated with",
    "treats",
    "has symptom",
    "causes",
    "side effect",
    "contraindication",
    "synergistic interaction",
    "off-label use",
    "linked to",
    "indication",
    "biomarker for",
    "complicates",
    "interacts with",
    "contraindicated with"
}

def extract_k_hop_subgraph(G: nx.Graph, center_nodes: Set[str], k: int = K_HOP) -> nx.Graph:
    print(f"🔍 Extracting {k}-hop subgraph from {len(center_nodes)} center node(s)...")
    nodes_to_include = set()
    for node in center_nodes:
        if node in G:
            nearby = nx.single_source_shortest_path_length(G, node, cutoff=k).keys()
            nodes_to_include.update(nearby)
    print(f"✅ Candidate subgraph has {len(nodes_to_include)} nodes.")

    full_subgraph = G.subgraph(nodes_to_include).copy()

    # Filter by relevant relation types
    edges_to_keep = [
        (u, v) for u, v, d in full_subgraph.edges(data=True)
        if any(rel in d.get("display_relation", "").lower() for rel in RELEVANT_DISPLAY_RELATIONS)
    ]

    filtered_subgraph = nx.Graph()
    for node in nodes_to_include:
        filtered_subgraph.add_node(node, **G.nodes[node])

    for u, v in edges_to_keep:
        filtered_subgraph.add_edge(u, v, **G.edges[u, v])

    print(f"🎯 Filtered subgraph has {filtered_subgraph.number_of_nodes()} nodes and {filtered_subgraph.number_of_edges()} edges.")
    return filtered_subgraph


def get_patient_matched_nodes(patient_data: dict, G: nx.Graph) -> Set[str]:
    """Match patient diagnoses/meds/symptoms to KG node names"""
    print("🧬 Matching patient data to graph node names...")
    matched = set()
    for key in ['conditions', 'medications', 'symptoms']:
        terms = patient_data.get(key, [])
        for term in terms:
            for node in G.nodes:
                node_name = G.nodes[node].get('node_name', '').lower()
                if term.lower() in node_name:
                    matched.add(node)
    return matched


def get_subgraph_for_patient(patient_data: dict, G: nx.Graph) -> nx.Graph:
    print("🚦 Starting subgraph generation for a patient.")

    center_nodes = get_patient_matched_nodes(patient_data, G)
    if not center_nodes:
        print("⚠️ No matching nodes found for patient. Returning empty subgraph.")
        return nx.Graph()

    subgraph = extract_k_hop_subgraph(G, center_nodes, k=K_HOP)
    print(f"✅ Subgraph generated with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")
    return subgraph

