from itertools import product
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
import pytest
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.csgraph import (
def test_explicit_zero_causes_warning():
    with pytest.warns(UserWarning):
        biadjacency_matrix = csr_matrix(((2, 0, 3), (0, 1, 1), (0, 2, 3)))
        min_weight_full_bipartite_matching(biadjacency_matrix)