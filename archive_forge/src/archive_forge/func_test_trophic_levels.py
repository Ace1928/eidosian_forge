import pytest
import networkx as nx
def test_trophic_levels():
    """Trivial example"""
    G = nx.DiGraph()
    G.add_edge('a', 'b')
    G.add_edge('b', 'c')
    d = nx.trophic_levels(G)
    assert d == {'a': 1, 'b': 2, 'c': 3}