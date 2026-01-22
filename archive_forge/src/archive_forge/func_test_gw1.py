import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
from networkx.algorithms.flow import (
@pytest.mark.slow
def test_gw1(self):
    G = read_graph('gw1')
    s = 1
    t = len(G)
    R = build_residual_network(G, 'capacity')
    kwargs = {'residual': R}
    for flow_func in flow_funcs:
        validate_flows(G, s, t, 1202018, flow_func(G, s, t, **kwargs), flow_func)