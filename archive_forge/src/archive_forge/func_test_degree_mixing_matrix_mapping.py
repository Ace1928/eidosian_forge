import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_degree_mixing_matrix_mapping(self):
    a_result = np.array([[6.0, 1.0], [1.0, 0.0]])
    mapping = {0.5: 1, 1.5: 0}
    a = nx.degree_mixing_matrix(self.W, weight='weight', normalized=False, mapping=mapping)
    np.testing.assert_equal(a, a_result)