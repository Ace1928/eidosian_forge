from itertools import product
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
import pytest
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.csgraph import (
def linear_sum_assignment_assertions(solver, array_type, sign, test_case):
    cost_matrix, expected_cost = test_case
    maximize = sign == -1
    cost_matrix = sign * array_type(cost_matrix)
    expected_cost = sign * np.array(expected_cost)
    row_ind, col_ind = solver(cost_matrix, maximize=maximize)
    assert_array_equal(row_ind, np.sort(row_ind))
    assert_array_equal(expected_cost, np.array(cost_matrix[row_ind, col_ind]).flatten())
    cost_matrix = cost_matrix.T
    row_ind, col_ind = solver(cost_matrix, maximize=maximize)
    assert_array_equal(row_ind, np.sort(row_ind))
    assert_array_equal(np.sort(expected_cost), np.sort(np.array(cost_matrix[row_ind, col_ind])).flatten())