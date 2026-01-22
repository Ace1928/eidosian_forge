import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_flow_func_parameters(self):
    G = self.G
    fv = 3.0
    for interface_func in interface_funcs:
        for flow_func in flow_funcs:
            errmsg = f'Assertion failed in function: {flow_func.__name__} in interface {interface_func.__name__}'
            result = interface_func(G, 'x', 'y', flow_func=flow_func)
            if interface_func in max_min_funcs:
                result = result[0]
            assert fv == result, errmsg