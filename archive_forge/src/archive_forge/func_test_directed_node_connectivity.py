import pytest
import networkx as nx
from networkx.algorithms import approximation as approx
def test_directed_node_connectivity():
    G = nx.cycle_graph(10, create_using=nx.DiGraph())
    D = nx.cycle_graph(10).to_directed()
    assert 1 == approx.node_connectivity(G)
    assert 1 == approx.node_connectivity(G, 1, 4)
    assert 2 == approx.node_connectivity(D)
    assert 2 == approx.node_connectivity(D, 1, 4)