import numpy as np
from numpy.testing import assert_array_equal
import pytest
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.sparse.csgraph._flow import (
@pytest.mark.parametrize('method', methods)
def test_add_reverse_edges_large_graph(method):
    n = 100000
    indices = np.arange(1, n)
    indptr = np.array(list(range(n)) + [n - 1])
    data = np.ones(n - 1, dtype=np.int32)
    graph = csr_matrix((data, indices, indptr), shape=(n, n))
    res = maximum_flow(graph, 0, n - 1, method=method)
    assert res.flow_value == 1
    expected_flow = graph - graph.transpose()
    assert_array_equal(res.flow.data, expected_flow.data)
    assert_array_equal(res.flow.indices, expected_flow.indices)
    assert_array_equal(res.flow.indptr, expected_flow.indptr)