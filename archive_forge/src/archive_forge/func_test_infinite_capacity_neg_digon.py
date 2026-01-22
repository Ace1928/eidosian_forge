import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
def test_infinite_capacity_neg_digon(self):
    """An infinite capacity negative cost digon results in an unbounded
        instance."""
    nodes = [(1, {}), (2, {'demand': -4}), (3, {'demand': 4})]
    edges = [(1, 2, {'weight': -600}), (2, 1, {'weight': 0}), (2, 3, {'capacity': 5, 'weight': 714285}), (3, 2, {'capacity': 2, 'weight': 0})]
    G = nx.DiGraph(edges)
    G.add_nodes_from(nodes)
    pytest.raises(nx.NetworkXUnbounded, nx.network_simplex, G)
    pytest.raises(nx.NetworkXUnbounded, nx.capacity_scaling, G)