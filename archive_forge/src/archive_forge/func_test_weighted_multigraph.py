import pytest
import networkx as nx
def test_weighted_multigraph(self):
    """Edge betweenness centrality: weighted multigraph"""
    eList = [(0, 1, 5), (0, 1, 4), (0, 2, 4), (0, 3, 3), (0, 3, 3), (0, 4, 2), (1, 2, 4), (1, 3, 1), (1, 3, 2), (1, 4, 3), (1, 4, 4), (2, 4, 5), (3, 4, 4), (3, 4, 4)]
    G = nx.MultiGraph()
    G.add_weighted_edges_from(eList)
    b = nx.edge_betweenness_centrality(G, weight='weight', normalized=False)
    b_answer = {(0, 1, 0): 0.0, (0, 1, 1): 0.5, (0, 2, 0): 1.0, (0, 3, 0): 0.75, (0, 3, 1): 0.75, (0, 4, 0): 1.0, (1, 2, 0): 2.0, (1, 3, 0): 3.0, (1, 3, 1): 0.0, (1, 4, 0): 1.5, (1, 4, 1): 0.0, (2, 4, 0): 1.0, (3, 4, 0): 0.25, (3, 4, 1): 0.25}
    for n in sorted(G.edges(keys=True)):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)