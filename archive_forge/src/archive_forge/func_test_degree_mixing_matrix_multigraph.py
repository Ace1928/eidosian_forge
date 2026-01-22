import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_degree_mixing_matrix_multigraph(self):
    a_result = np.array([[0, 1, 0], [1, 0, 3], [0, 3, 0]])
    a = nx.degree_mixing_matrix(self.M, normalized=False)
    np.testing.assert_equal(a, a_result)
    a = nx.degree_mixing_matrix(self.M)
    np.testing.assert_equal(a, a_result / a_result.sum())