import pytest
import networkx as nx
def test_newman_watts_strogatz_zero_probability():
    G = nx.newman_watts_strogatz_graph(10, 2, 0.0, seed=42)
    assert len(G) == 10
    assert G.number_of_edges() == 10