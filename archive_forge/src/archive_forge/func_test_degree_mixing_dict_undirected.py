import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_degree_mixing_dict_undirected(self):
    d = nx.degree_mixing_dict(self.P4)
    d_result = {1: {2: 2}, 2: {1: 2, 2: 2}}
    assert d == d_result