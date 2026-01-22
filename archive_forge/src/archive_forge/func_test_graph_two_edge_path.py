import pytest
import networkx as nx
def test_graph_two_edge_path(self):
    G = nx.path_graph(3)
    min_cover = nx.min_edge_cover(G)
    assert len(min_cover) == 2
    for u, v in G.edges:
        assert (u, v) in min_cover or (v, u) in min_cover