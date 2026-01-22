import pytest
import networkx as nx
def test_G4(self):
    """Weighted betweenness centrality: G4"""
    G = nx.MultiDiGraph()
    G.add_weighted_edges_from([('s', 'u', 10), ('s', 'x', 5), ('s', 'x', 6), ('u', 'v', 1), ('u', 'x', 2), ('v', 'y', 1), ('v', 'y', 1), ('x', 'u', 3), ('x', 'v', 5), ('x', 'y', 2), ('x', 'y', 3), ('y', 's', 7), ('y', 'v', 6), ('y', 'v', 6)])
    b_answer = {'y': 5.0, 'x': 5.0, 's': 4.0, 'u': 2.0, 'v': 2.0}
    b = nx.betweenness_centrality(G, weight='weight', normalized=False)
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=1e-07)