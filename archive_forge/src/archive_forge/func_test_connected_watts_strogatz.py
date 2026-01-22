import pytest
import networkx as nx
def test_connected_watts_strogatz():
    G = nx.connected_watts_strogatz_graph(10, 2, 0.1, tries=10, seed=42)
    assert len(G) == 10
    assert G.number_of_edges() == 10