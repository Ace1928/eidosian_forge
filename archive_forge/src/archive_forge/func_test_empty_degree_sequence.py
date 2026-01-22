import pytest
import networkx as nx
def test_empty_degree_sequence(self):
    """Tests that an empty degree sequence yields the null graph."""
    G = nx.configuration_model([])
    assert len(G) == 0