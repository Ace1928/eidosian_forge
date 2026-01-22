import pytest
import networkx as nx
def test_P5_multiple_target(self):
    """Edge betweenness subset centrality: P5 multiple target"""
    G = nx.Graph()
    nx.add_path(G, range(5))
    b_answer = dict.fromkeys(G.edges(), 0)
    b_answer[0, 1] = b_answer[1, 2] = b_answer[2, 3] = 1
    b_answer[3, 4] = 0.5
    b = nx.edge_betweenness_centrality_subset(G, sources=[0], targets=[3, 4], weight=None)
    for n in sorted(G.edges()):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)