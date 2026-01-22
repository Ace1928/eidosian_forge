import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import (
def test_complete_graphs():
    for n in range(5, 20, 5):
        for flow_func in flow_funcs:
            G = nx.complete_graph(n)
            errmsg = f'Assertion failed in function: {flow_func.__name__}'
            assert n - 1 == nx.node_connectivity(G, flow_func=flow_func), errmsg
            assert n - 1 == nx.node_connectivity(G.to_directed(), flow_func=flow_func), errmsg
            assert n - 1 == nx.edge_connectivity(G, flow_func=flow_func), errmsg
            assert n - 1 == nx.edge_connectivity(G.to_directed(), flow_func=flow_func), errmsg