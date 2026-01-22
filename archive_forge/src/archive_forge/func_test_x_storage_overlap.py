import pytest
import numpy as np
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.optimize._differentiable_functions import (ScalarFunction,
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.optimize._hessian_update_strategy import BFGS
def test_x_storage_overlap(self):
    ex = ExVectorialFunction()
    x0 = np.array([1.0, 0.0])
    vf = VectorFunction(ex.fun, x0, '3-point', ex.hess, None, None, (-np.inf, np.inf), None)
    assert x0 is not vf.x
    assert_equal(vf.fun(x0), ex.fun(x0))
    assert x0 is not vf.x
    x0[0] = 2.0
    assert_equal(vf.fun(x0), ex.fun(x0))
    assert x0 is not vf.x
    x0[0] = 1.0
    assert_equal(vf.fun(x0), ex.fun(x0))
    assert x0 is not vf.x
    hess = BFGS()
    x0 = np.array([1.0, 0.0])
    vf = VectorFunction(ex.fun, x0, '3-point', hess, None, None, (-np.inf, np.inf), None)
    with pytest.warns(UserWarning):
        assert x0 is not vf.x
        assert_equal(vf.fun(x0), ex.fun(x0))
        assert x0 is not vf.x
        x0[0] = 2.0
        assert_equal(vf.fun(x0), ex.fun(x0))
        assert x0 is not vf.x
        x0[0] = 1.0
        assert_equal(vf.fun(x0), ex.fun(x0))
        assert x0 is not vf.x