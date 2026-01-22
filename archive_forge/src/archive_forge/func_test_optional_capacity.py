import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_optional_capacity(self):
    G = nx.DiGraph()
    G.add_edge('x', 'a', spam=3.0)
    G.add_edge('x', 'b', spam=1.0)
    G.add_edge('a', 'c', spam=3.0)
    G.add_edge('b', 'c', spam=5.0)
    G.add_edge('b', 'd', spam=4.0)
    G.add_edge('d', 'e', spam=2.0)
    G.add_edge('c', 'y', spam=2.0)
    G.add_edge('e', 'y', spam=3.0)
    solnFlows = {'x': {'a': 2.0, 'b': 1.0}, 'a': {'c': 2.0}, 'b': {'c': 0, 'd': 1.0}, 'c': {'y': 2.0}, 'd': {'e': 1.0}, 'e': {'y': 1.0}, 'y': {}}
    solnValue = 3.0
    s = 'x'
    t = 'y'
    compare_flows_and_cuts(G, s, t, solnFlows, solnValue, capacity='spam')