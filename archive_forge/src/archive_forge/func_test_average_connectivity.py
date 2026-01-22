import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import (
def test_average_connectivity():
    G1 = nx.path_graph(3)
    G1.add_edges_from([(1, 3), (1, 4)])
    G2 = nx.path_graph(3)
    G2.add_edges_from([(1, 3), (1, 4), (0, 3), (0, 4), (3, 4)])
    G3 = nx.Graph()
    for flow_func in flow_funcs:
        kwargs = {'flow_func': flow_func}
        errmsg = f'Assertion failed in function: {flow_func.__name__}'
        assert nx.average_node_connectivity(G1, **kwargs) == 1, errmsg
        assert nx.average_node_connectivity(G2, **kwargs) == 2.2, errmsg
        assert nx.average_node_connectivity(G3, **kwargs) == 0, errmsg