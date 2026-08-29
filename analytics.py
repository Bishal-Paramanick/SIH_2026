"""
analytics.py
Core graph analytics: centrality, community detection, anomaly detection.

All functions take a networkx graph (MultiDiGraph) and return plain
dicts/lists so they're easy to JSON-serialize for Person 4's API layer.
"""

import networkx as nx
from collections import defaultdict
from datetime import datetime


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------

def compute_centrality(G: nx.MultiDiGraph) -> dict:
    """Returns per-node centrality scores, all normalized to [0, 1]."""
    # networkx betweenness/pagerank on MultiDiGraph can be slow/inconsistent;
    # collapse to a simple weighted DiGraph for these calculations.
    simple = _collapse_to_weighted_digraph(G)

    degree = nx.degree_centrality(simple)
    betweenness = nx.betweenness_centrality(simple, weight="weight")
    try:
        pagerank = nx.pagerank(simple, weight="weight")
    except nx.PowerIterationFailedConvergence:
        pagerank = {n: 0.0 for n in simple.nodes()}

    result = {}
    for node in simple.nodes():
        result[node] = {
            "degree_centrality": round(degree.get(node, 0.0), 4),
            "betweenness_centrality": round(betweenness.get(node, 0.0), 4),
            "pagerank": round(pagerank.get(node, 0.0), 4),
        }
    return result


def _collapse_to_weighted_digraph(G: nx.MultiDiGraph) -> nx.DiGraph:
    """Multiple edges between the same pair (e.g. 18 calls) become one
    edge with weight = count, so a burst of calls correctly increases
    that connection's importance in centrality math."""
    simple = nx.DiGraph()
    simple.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        if simple.has_edge(u, v):
            simple[u][v]["weight"] += 1
        else:
            simple.add_edge(u, v, weight=1)
    return simple


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------

def detect_communities(G: nx.MultiDiGraph) -> dict:
    """Returns {node: community_id}. Louvain needs an undirected simple graph."""
    undirected = nx.Graph()
    undirected.add_nodes_from(G.nodes())
    for u, v in G.edges():
        if undirected.has_edge(u, v):
            undirected[u][v]["weight"] += 1
        else:
            undirected.add_edge(u, v, weight=1)

    communities = nx.algorithms.community.louvain_communities(undirected, weight="weight", seed=42)

    node_to_community = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx
    return node_to_community


# ---------------------------------------------------------------------------
# Anomaly detection (rule-based -- deliberately simple for hackathon scope)
# ---------------------------------------------------------------------------

def detect_circular_transactions(G: nx.MultiDiGraph, max_cycle_len: int = 4) -> list:
    """Finds short cycles of TRANSACTED_WITH edges -- classic laundering pattern
    (A pays B, B pays C, C pays back A)."""
    txn_graph = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if data.get("edge_type") == "TRANSACTED_WITH":
            txn_graph.add_edge(u, v)

    cycles = []
    for cycle in nx.simple_cycles(txn_graph, length_bound=max_cycle_len):
        if len(cycle) >= 2:
            cycles.append(cycle)
    return cycles


def detect_call_bursts(G: nx.MultiDiGraph, threshold_per_day: int = 10) -> list:
    """Flags (source, target) pairs with more than `threshold_per_day` CALLED
    edges on the same calendar day."""
    calls_by_pair_day = defaultdict(int)

    for u, v, data in G.edges(data=True):
        if data.get("edge_type") != "CALLED":
            continue
        ts = data.get("timestamp")
        if not ts:
            continue
        ts = ts.replace("Z", "+00:00")  # handle UTC 'Z' suffix Person 1/2's data uses
        day = datetime.fromisoformat(ts).date().isoformat()
        calls_by_pair_day[(u, v, day)] += 1

    flagged = []
    for (u, v, day), count in calls_by_pair_day.items():
        if count > threshold_per_day:
            flagged.append({"source": u, "target": v, "date": day, "call_count": count})
    return flagged


def detect_cross_case_entities(G: nx.MultiDiGraph) -> dict:
    """Returns {node: num_distinct_cases} for entities appearing in 2+ cases --
    a strong investigative signal on its own. Works off case_ids regardless
    of node type, so it doesn't actually need the type check -- but we guard
    on it anyway since only Person nodes are expected to carry case_ids."""
    result = {}
    for node, data in G.nodes(data=True):
        case_ids = data.get("case_ids", [])
        if len(case_ids) >= 2:
            result[node] = len(case_ids)
    return result


if __name__ == "__main__":
    from graph_loader import load_graph
    import json

    G = load_graph()

    print("\n=== Centrality ===")
    print(json.dumps(compute_centrality(G), indent=2))

    print("\n=== Communities ===")
    print(json.dumps(detect_communities(G), indent=2))

    print("\n=== Circular transactions ===")
    print(json.dumps(detect_circular_transactions(G), indent=2))

    print("\n=== Call bursts (>10/day) ===")
    print(json.dumps(detect_call_bursts(G), indent=2))

    print("\n=== Cross-case entities ===")
    print(json.dumps(detect_cross_case_entities(G), indent=2))
