import numpy as np
import pytest
import cvxpy as cvx
import cvxpy.problems.iterative as iterative
import cvxpy.settings as s
from cvxpy.lin_ops.tree_mat import prune_constants
from cvxpy.tests.base_test import BaseTest
def mat_from_func(self, func, rows, cols):
    """Convert a multiplier function to a matrix.
        """
    test_vec = np.zeros(cols)
    result = np.zeros(rows)
    matrix = np.zeros((rows, cols))
    for i in range(cols):
        test_vec[i] = 1.0
        func(test_vec, result)
        matrix[:, i] = result
        test_vec *= 0
        result *= 0
    return matrix