import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_digraph4(self):
    G = nx.DiGraph()
    G.add_edge('x', 'a', capacity=3.0)
    G.add_edge('x', 'b', capacity=1.0)
    G.add_edge('a', 'c', capacity=3.0)
    G.add_edge('b', 'c', capacity=5.0)
    G.add_edge('b', 'd', capacity=4.0)
    G.add_edge('d', 'e', capacity=2.0)
    G.add_edge('c', 'y', capacity=2.0)
    G.add_edge('e', 'y', capacity=3.0)
    H = {'x': {'a': 2.0, 'b': 1.0}, 'a': {'c': 2.0}, 'b': {'c': 0, 'd': 1.0}, 'c': {'y': 2.0}, 'd': {'e': 1.0}, 'e': {'y': 1.0}, 'y': {}}
    compare_flows_and_cuts(G, 'x', 'y', H, 3.0)