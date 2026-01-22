import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
@pytest.fixture
def simple_flow_graph():
    G = nx.DiGraph()
    G.add_node('a', demand=0)
    G.add_node('b', demand=-5)
    G.add_node('c', demand=50000000)
    G.add_node('d', demand=-49999995)
    G.add_edge('a', 'b', weight=3, capacity=4)
    G.add_edge('a', 'c', weight=6, capacity=10)
    G.add_edge('b', 'd', weight=1, capacity=9)
    G.add_edge('c', 'd', weight=2, capacity=5)
    return G