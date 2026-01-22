import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_node_degree_xy_selfloop(self):
    xy = sorted(nx.node_degree_xy(self.S))
    xy_result = sorted([(2, 2), (2, 2)])
    assert xy == xy_result