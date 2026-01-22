import pytest
import networkx as nx
def test_P3_normalized(self):
    """Weighted betweenness centrality: P3 normalized"""
    G = nx.path_graph(3)
    b = nx.betweenness_centrality(G, weight='weight', normalized=True)
    b_answer = {0: 0.0, 1: 1.0, 2: 0.0}
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)