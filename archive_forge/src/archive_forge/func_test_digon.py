import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
def test_digon(self):
    """Check if digons are handled properly. Taken from ticket
        #618 by arv."""
    nodes = [(1, {}), (2, {'demand': -4}), (3, {'demand': 4})]
    edges = [(1, 2, {'capacity': 3, 'weight': 600000}), (2, 1, {'capacity': 2, 'weight': 0}), (2, 3, {'capacity': 5, 'weight': 714285}), (3, 2, {'capacity': 2, 'weight': 0})]
    G = nx.DiGraph(edges)
    G.add_nodes_from(nodes)
    flowCost, H = nx.network_simplex(G)
    soln = {1: {2: 0}, 2: {1: 0, 3: 4}, 3: {2: 0}}
    assert flowCost == 2857140
    assert nx.min_cost_flow_cost(G) == 2857140
    assert H == soln
    assert nx.min_cost_flow(G) == soln
    assert nx.cost_of_flow(G, H) == 2857140
    flowCost, H = nx.capacity_scaling(G)
    assert flowCost == 2857140
    assert H == soln
    assert nx.cost_of_flow(G, H) == 2857140