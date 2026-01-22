import pytest
import networkx as nx
def test_directed_path_normalized(self):
    """Betweenness centrality: directed path normalized"""
    G = nx.DiGraph()
    nx.add_path(G, [0, 1, 2])
    b = nx.betweenness_centrality(G, weight=None, normalized=True)
    b_answer = {0: 0.0, 1: 0.5, 2: 0.0}
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)