from numpy.testing import assert_array_equal
import pytest
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import random
from scipy.sparse._sputils import matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.sparse.csgraph.tests.test_matching import (
def test_constant_cost_matrix():
    n = 8
    C = np.ones((n, n))
    row_ind, col_ind = linear_sum_assignment(C)
    assert_array_equal(row_ind, np.arange(n))
    assert_array_equal(col_ind, np.arange(n))