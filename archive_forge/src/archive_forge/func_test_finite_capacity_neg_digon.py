import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
def test_finite_capacity_neg_digon(self):
    """The digon should receive the maximum amount of flow it can handle.
        Taken from ticket #749 by @chuongdo."""
    G = nx.DiGraph()
    G.add_edge('a', 'b', capacity=1, weight=-1)
    G.add_edge('b', 'a', capacity=1, weight=-1)
    min_cost = -2
    assert nx.min_cost_flow_cost(G) == min_cost
    flowCost, H = nx.capacity_scaling(G)
    assert flowCost == -2
    assert H == {'a': {'b': 1}, 'b': {'a': 1}}
    assert nx.cost_of_flow(G, H) == -2