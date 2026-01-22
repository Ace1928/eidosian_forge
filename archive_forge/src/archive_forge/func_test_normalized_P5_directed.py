import pytest
import networkx as nx
def test_normalized_P5_directed(self):
    """Edge betweenness subset centrality: Normalized Directed P5"""
    G = nx.DiGraph()
    nx.add_path(G, range(5))
    b_answer = dict.fromkeys(G.edges(), 0)
    b_answer[0, 1] = b_answer[1, 2] = b_answer[2, 3] = 0.05
    b = nx.edge_betweenness_centrality_subset(G, sources=[0], targets=[3], normalized=True, weight=None)
    for n in G.edges():
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)