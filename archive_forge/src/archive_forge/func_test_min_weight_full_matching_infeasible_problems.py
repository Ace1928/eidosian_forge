from itertools import product
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
import pytest
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.csgraph import (
@pytest.mark.parametrize('biadjacency_matrix', [[[1, 1, 1], [1, 0, 0], [1, 0, 0]], [[1, 1, 1], [0, 0, 1], [0, 0, 1]], [[1, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]], [[1, 0, 0], [2, 0, 0]], [[0, 1, 0], [0, 2, 0]], [[1, 0], [2, 0], [5, 0]]])
def test_min_weight_full_matching_infeasible_problems(biadjacency_matrix):
    with pytest.raises(ValueError):
        min_weight_full_bipartite_matching(csr_matrix(biadjacency_matrix))