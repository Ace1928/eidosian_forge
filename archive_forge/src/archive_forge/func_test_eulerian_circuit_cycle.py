import collections
import pytest
import networkx as nx
def test_eulerian_circuit_cycle(self):
    G = nx.cycle_graph(4)
    edges = list(nx.eulerian_circuit(G, source=0))
    nodes = [u for u, v in edges]
    assert nodes == [0, 3, 2, 1]
    assert edges == [(0, 3), (3, 2), (2, 1), (1, 0)]
    edges = list(nx.eulerian_circuit(G, source=1))
    nodes = [u for u, v in edges]
    assert nodes == [1, 2, 3, 0]
    assert edges == [(1, 2), (2, 3), (3, 0), (0, 1)]
    G = nx.complete_graph(3)
    edges = list(nx.eulerian_circuit(G, source=0))
    nodes = [u for u, v in edges]
    assert nodes == [0, 2, 1]
    assert edges == [(0, 2), (2, 1), (1, 0)]
    edges = list(nx.eulerian_circuit(G, source=1))
    nodes = [u for u, v in edges]
    assert nodes == [1, 2, 0]
    assert edges == [(1, 2), (2, 0), (0, 1)]