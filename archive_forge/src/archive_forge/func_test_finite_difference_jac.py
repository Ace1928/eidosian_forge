import pytest
import numpy as np
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.optimize._differentiable_functions import (ScalarFunction,
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.optimize._hessian_update_strategy import BFGS
def test_finite_difference_jac(self):
    ex = ExVectorialFunction()
    nfev = 0
    njev = 0
    x0 = [1.0, 0.0]
    analit = VectorFunction(ex.fun, x0, ex.jac, ex.hess, None, None, (-np.inf, np.inf), None)
    nfev += 1
    njev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev, njev)
    approx = VectorFunction(ex.fun, x0, '2-point', ex.hess, None, None, (-np.inf, np.inf), None)
    nfev += 3
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev + approx.njev, njev)
    assert_array_equal(analit.f, approx.f)
    assert_array_almost_equal(analit.J, approx.J)
    x = [10, 0.3]
    f_analit = analit.fun(x)
    J_analit = analit.jac(x)
    nfev += 1
    njev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev + approx.njev, njev)
    f_approx = approx.fun(x)
    J_approx = approx.jac(x)
    nfev += 3
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev + approx.njev, njev)
    assert_array_almost_equal(f_analit, f_approx)
    assert_array_almost_equal(J_analit, J_approx, decimal=4)
    x = [2.0, 1.0]
    J_analit = analit.jac(x)
    njev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev + approx.njev, njev)
    J_approx = approx.jac(x)
    nfev += 3
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev + approx.njev, njev)
    assert_array_almost_equal(J_analit, J_approx)
    x = [2.5, 0.3]
    f_analit = analit.fun(x)
    J_analit = analit.jac(x)
    nfev += 1
    njev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev + approx.njev, njev)
    f_approx = approx.fun(x)
    J_approx = approx.jac(x)
    nfev += 3
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev + approx.njev, njev)
    assert_array_almost_equal(f_analit, f_approx)
    assert_array_almost_equal(J_analit, J_approx)
    x = [2, 0.3]
    f_analit = analit.fun(x)
    J_analit = analit.jac(x)
    nfev += 1
    njev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev + approx.njev, njev)
    f_approx = approx.fun(x)
    J_approx = approx.jac(x)
    nfev += 3
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(ex.njev, njev)
    assert_array_equal(analit.njev + approx.njev, njev)
    assert_array_almost_equal(f_analit, f_approx)
    assert_array_almost_equal(J_analit, J_approx)