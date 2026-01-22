import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
def test_no_constraints(self):
    prob = Rosenbrock()
    result = minimize(prob.fun, prob.x0, method='trust-constr', jac=prob.grad, hess=prob.hess)
    result1 = minimize(prob.fun, prob.x0, method='L-BFGS-B', jac='2-point')
    result2 = minimize(prob.fun, prob.x0, method='L-BFGS-B', jac='3-point')
    assert_array_almost_equal(result.x, prob.x_opt, decimal=5)
    assert_array_almost_equal(result1.x, prob.x_opt, decimal=5)
    assert_array_almost_equal(result2.x, prob.x_opt, decimal=5)