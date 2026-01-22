import pytest
import networkx as nx
def test_generalized_degree(self):
    G = nx.Graph()
    assert nx.generalized_degree(G) == {}