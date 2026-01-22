import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_degree_mixing_dict_directed(self):
    d = nx.degree_mixing_dict(self.D)
    print(d)
    d_result = {1: {3: 2}, 2: {1: 1, 3: 1}, 3: {}}
    assert d == d_result