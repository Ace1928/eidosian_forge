import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
def test_degree_mixing_matrix_undirected(self):
    a_result = np.array([[0, 2], [2, 2]])
    a = nx.degree_mixing_matrix(self.P4, normalized=False)
    np.testing.assert_equal(a, a_result)
    a = nx.degree_mixing_matrix(self.P4)
    np.testing.assert_equal(a, a_result / a_result.sum())