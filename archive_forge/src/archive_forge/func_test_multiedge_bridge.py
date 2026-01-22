import pytest
import networkx as nx
def test_multiedge_bridge(self):
    edges = [(0, 1), (0, 2), (1, 2), (1, 2), (2, 3), (3, 4), (3, 4)]
    G = nx.MultiGraph(edges)
    assert nx.has_bridges(G)
    G.add_edges_from([(0, 1), (0, 2), (2, 3)])
    assert not nx.has_bridges(G)