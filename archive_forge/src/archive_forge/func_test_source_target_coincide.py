import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_source_target_coincide(self):
    G = nx.Graph()
    G.add_node(0)
    for flow_func in all_funcs:
        pytest.raises(nx.NetworkXError, flow_func, G, 0, 0)