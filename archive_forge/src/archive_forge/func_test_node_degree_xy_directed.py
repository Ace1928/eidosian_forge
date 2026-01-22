import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_node_degree_xy_directed(self):
    xy = sorted(nx.node_degree_xy(self.D))
    xy_result = sorted([(2, 1), (2, 3), (1, 3), (1, 3)])
    assert xy == xy_result