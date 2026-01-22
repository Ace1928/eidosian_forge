import pytest
import networkx as nx
def test_diamond_multi_path(self):
    """Edge betweenness subset centrality: Diamond Multi Path"""
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (1, 10), (10, 11), (11, 12), (12, 9), (2, 6), (3, 6), (4, 6), (5, 7), (7, 8), (6, 8), (8, 9)])
    b_answer = dict.fromkeys(G.edges(), 0)
    b_answer[8, 9] = 0.4
    b_answer[6, 8] = b_answer[7, 8] = 0.2
    b_answer[2, 6] = b_answer[3, 6] = b_answer[4, 6] = 0.2 / 3.0
    b_answer[1, 2] = b_answer[1, 3] = b_answer[1, 4] = 0.2 / 3.0
    b_answer[5, 7] = 0.2
    b_answer[1, 5] = 0.2
    b_answer[9, 12] = 0.1
    b_answer[11, 12] = b_answer[10, 11] = b_answer[1, 10] = 0.1
    b = nx.edge_betweenness_centrality_subset(G, sources=[1], targets=[9], weight=None)
    for n in G.edges():
        sort_n = tuple(sorted(n))
        assert b[n] == pytest.approx(b_answer[sort_n], abs=1e-07)