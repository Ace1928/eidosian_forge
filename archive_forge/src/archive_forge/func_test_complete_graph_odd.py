import pytest
import networkx as nx
def test_complete_graph_odd(self):
    G = nx.complete_graph(11)
    min_cover = nx.min_edge_cover(G)
    assert nx.is_edge_cover(G, min_cover)
    assert len(min_cover) == 6