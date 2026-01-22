import pytest
import networkx as nx
def test_box(self):
    """Edge betweenness subset centrality: box"""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    b_answer = dict.fromkeys(G.edges(), 0)
    b_answer[0, 1] = b_answer[0, 2] = 0.25
    b_answer[1, 3] = b_answer[2, 3] = 0.25
    b = nx.edge_betweenness_centrality_subset(G, sources=[0], targets=[3], weight=None)
    for n in sorted(G.edges()):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)