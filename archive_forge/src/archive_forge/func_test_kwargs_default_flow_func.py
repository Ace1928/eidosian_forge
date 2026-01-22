import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_kwargs_default_flow_func(self):
    G = self.H
    for interface_func in interface_funcs:
        pytest.raises(nx.NetworkXError, interface_func, G, 0, 1, global_relabel_freq=2)