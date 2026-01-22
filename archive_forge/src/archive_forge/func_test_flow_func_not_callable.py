import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_flow_func_not_callable(self):
    elements = ['this_should_be_callable', 10, {1, 2, 3}]
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (2, 3, 1)], weight='capacity')
    for flow_func in interface_funcs:
        for element in elements:
            pytest.raises(nx.NetworkXError, flow_func, G, 0, 1, flow_func=element)
            pytest.raises(nx.NetworkXError, flow_func, G, 0, 1, flow_func=element)