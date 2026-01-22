import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
def test_default_jac_and_hess(self):

    def fun(x):
        return (x - 1) ** 2
    bounds = [(-2, 2)]
    res = minimize(fun, x0=[-1.5], bounds=bounds, method='trust-constr')
    assert_array_almost_equal(res.x, 1, decimal=5)