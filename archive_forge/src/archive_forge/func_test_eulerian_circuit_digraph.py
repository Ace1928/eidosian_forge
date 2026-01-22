import collections
import pytest
import networkx as nx
def test_eulerian_circuit_digraph(self):
    G = nx.DiGraph()
    nx.add_cycle(G, [0, 1, 2, 3])
    edges = list(nx.eulerian_circuit(G, source=0))
    nodes = [u for u, v in edges]
    assert nodes == [0, 1, 2, 3]
    assert edges == [(0, 1), (1, 2), (2, 3), (3, 0)]
    edges = list(nx.eulerian_circuit(G, source=1))
    nodes = [u for u, v in edges]
    assert nodes == [1, 2, 3, 0]
    assert edges == [(1, 2), (2, 3), (3, 0), (0, 1)]