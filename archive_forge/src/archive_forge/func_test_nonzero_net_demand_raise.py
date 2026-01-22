import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
def test_nonzero_net_demand_raise(simple_flow_graph):
    G = simple_flow_graph
    nx.set_node_attributes(G, {'b': {'demand': -4}})
    pytest.raises(nx.NetworkXUnfeasible, nx.network_simplex, G)