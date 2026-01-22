import pytest
import networkx as nx
def test_graph_with_loop(self):
    G = nx.Graph()
    G.add_edge(1, 1)
    assert nx.is_edge_cover(G, {(1, 1)})