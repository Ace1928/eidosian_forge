import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_attribute_mixing_matrix_multigraph(self):
    mapping = {'one': 0, 'two': 1, 'red': 2, 'blue': 3}
    a_result = np.array([[4, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    a = nx.attribute_mixing_matrix(self.M, 'fish', mapping=mapping, normalized=False)
    np.testing.assert_equal(a, a_result)
    a = nx.attribute_mixing_matrix(self.M, 'fish', mapping=mapping)
    np.testing.assert_equal(a, a_result / a_result.sum())