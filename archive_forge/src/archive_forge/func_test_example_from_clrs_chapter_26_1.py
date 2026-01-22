import numpy as np
from numpy.testing import assert_array_equal
import pytest
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.sparse.csgraph._flow import (
@pytest.mark.parametrize('method', methods)
def test_example_from_clrs_chapter_26_1(method):
    graph = csr_matrix([[0, 16, 13, 0, 0, 0], [0, 0, 10, 12, 0, 0], [0, 4, 0, 0, 14, 0], [0, 0, 9, 0, 0, 20], [0, 0, 0, 7, 0, 4], [0, 0, 0, 0, 0, 0]])
    res = maximum_flow(graph, 0, 5, method=method)
    assert res.flow_value == 23
    expected_flow = np.array([[0, 12, 11, 0, 0, 0], [-12, 0, 0, 12, 0, 0], [-11, 0, 0, 0, 11, 0], [0, -12, 0, 0, -7, 19], [0, 0, -11, 7, 0, 4], [0, 0, 0, -19, -4, 0]])
    assert_array_equal(res.flow.toarray(), expected_flow)