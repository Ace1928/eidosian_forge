import pytest
import numpy as np
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.optimize._differentiable_functions import (ScalarFunction,
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.optimize._hessian_update_strategy import BFGS
def test_LinearVectorFunction_memoization():
    A = np.array([[-1, 2, 0], [0, 4, 2]])
    x0 = np.array([1, 2, -1])
    fun = LinearVectorFunction(A, x0, False)
    assert_array_equal(x0, fun.x)
    assert_array_equal(A.dot(x0), fun.f)
    x1 = np.array([-1, 3, 10])
    assert_array_equal(A, fun.jac(x1))
    assert_array_equal(x1, fun.x)
    assert_array_equal(A.dot(x0), fun.f)
    assert_array_equal(A.dot(x1), fun.fun(x1))
    assert_array_equal(A.dot(x1), fun.f)