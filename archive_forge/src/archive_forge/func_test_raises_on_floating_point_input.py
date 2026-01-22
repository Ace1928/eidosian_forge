import numpy as np
from numpy.testing import assert_array_equal
import pytest
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.sparse.csgraph._flow import (
def test_raises_on_floating_point_input():
    with pytest.raises(ValueError):
        graph = csr_matrix([[0, 1.5], [0, 0]], dtype=np.float64)
        maximum_flow(graph, 0, 1)
        maximum_flow(graph, 0, 1, method='edmonds_karp')