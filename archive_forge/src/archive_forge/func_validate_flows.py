import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
from networkx.algorithms.flow import (
def validate_flows(G, s, t, soln_value, R, flow_func):
    flow_value = R.graph['flow_value']
    flow_dict = build_flow_dict(G, R)
    errmsg = f'Assertion failed in function: {flow_func.__name__}'
    assert soln_value == flow_value, errmsg
    assert set(G) == set(flow_dict), errmsg
    for u in G:
        assert set(G[u]) == set(flow_dict[u]), errmsg
    excess = {u: 0 for u in flow_dict}
    for u in flow_dict:
        for v, flow in flow_dict[u].items():
            assert flow <= G[u][v].get('capacity', float('inf')), errmsg
            assert flow >= 0, errmsg
            excess[u] -= flow
            excess[v] += flow
    for u, exc in excess.items():
        if u == s:
            assert exc == -soln_value, errmsg
        elif u == t:
            assert exc == soln_value, errmsg
        else:
            assert exc == 0, errmsg