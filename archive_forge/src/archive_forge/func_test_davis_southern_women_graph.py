from itertools import combinations
import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_davis_southern_women_graph(self):
    G = nx.davis_southern_women_graph()
    nx.set_edge_attributes(G, 1, 'capacity')
    for flow_func in flow_funcs:
        T = nx.gomory_hu_tree(G, flow_func=flow_func)
        assert nx.is_tree(T)
        for u, v in combinations(G, 2):
            cut_value, edge = self.minimum_edge_weight(T, u, v)
            assert nx.minimum_cut_value(G, u, v) == cut_value