import pytest
import networkx as nx
def test_newman_watts_strogatz_nonzero_probability():
    G = nx.newman_watts_strogatz_graph(10, 4, 0.25, seed=42)
    assert len(G) == 10
    assert G.number_of_edges() >= 20