import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_degree_mixing_dict_multigraph(self):
    d = nx.degree_mixing_dict(self.M)
    d_result = {1: {2: 1}, 2: {1: 1, 3: 3}, 3: {2: 3}}
    assert d == d_result