from itertools import product
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
import pytest
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.csgraph import (
def test_maximum_bipartite_matching_empty_graph():
    graph = csr_matrix((0, 0))
    x = maximum_bipartite_matching(graph, perm_type='row')
    y = maximum_bipartite_matching(graph, perm_type='column')
    expected_matching = np.array([])
    assert_array_equal(expected_matching, x)
    assert_array_equal(expected_matching, y)