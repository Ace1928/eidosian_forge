import collections
import pytest
import networkx as nx
def test_has_eulerian_path_cyclic(self):
    assert nx.has_eulerian_path(nx.complete_graph(5))
    assert nx.has_eulerian_path(nx.complete_graph(7))
    assert nx.has_eulerian_path(nx.hypercube_graph(4))
    assert nx.has_eulerian_path(nx.hypercube_graph(6))