from itertools import product
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
import pytest
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.csgraph import (
def test_maximum_bipartite_matching_feasibility_of_result():
    data = np.ones(50, dtype=int)
    indices = [11, 12, 19, 22, 23, 5, 22, 3, 8, 10, 5, 6, 11, 12, 13, 5, 13, 14, 20, 22, 3, 15, 3, 13, 14, 11, 12, 19, 22, 23, 5, 22, 3, 8, 10, 5, 6, 11, 12, 13, 5, 13, 14, 20, 22, 3, 15, 3, 13, 14]
    indptr = [0, 5, 7, 10, 10, 15, 20, 22, 22, 23, 25, 30, 32, 35, 35, 40, 45, 47, 47, 48, 50]
    graph = csr_matrix((data, indices, indptr), shape=(20, 25))
    x = maximum_bipartite_matching(graph, perm_type='row')
    y = maximum_bipartite_matching(graph, perm_type='column')
    assert (x != -1).sum() == 13
    assert (y != -1).sum() == 13
    for u, v in zip(range(graph.shape[0]), y):
        if v != -1:
            assert graph[u, v]
    for u, v in zip(x, range(graph.shape[1])):
        if u != -1:
            assert graph[u, v]