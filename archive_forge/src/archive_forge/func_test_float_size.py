import pytest
import numpy as np
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.optimize._differentiable_functions import (ScalarFunction,
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.optimize._hessian_update_strategy import BFGS
def test_float_size(self):
    ex = ExVectorialFunction()
    x0 = np.array([1.0, 0.0]).astype(np.float32)
    vf = VectorFunction(ex.fun, x0, ex.jac, ex.hess, None, None, (-np.inf, np.inf), None)
    res = vf.fun(x0)
    assert res.dtype == np.float32
    res = vf.jac(x0)
    assert res.dtype == np.float32