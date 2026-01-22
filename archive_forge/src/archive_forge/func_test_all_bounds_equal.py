import itertools
import platform
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy import optimize
from scipy.optimize._minimize import Bounds, NonlinearConstraint
from scipy.optimize._minimize import (MINIMIZE_METHODS,
from scipy.optimize._linprog import LINPROG_METHODS
from scipy.optimize._root import ROOT_METHODS
from scipy.optimize._root_scalar import ROOT_SCALAR_METHODS
from scipy.optimize._qap import QUADRATIC_ASSIGNMENT_METHODS
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
from scipy.optimize._optimize import MemoizeJac, show_options, OptimizeResult
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.sparse import (coo_matrix, csc_matrix, csr_matrix, coo_array,
@pytest.mark.parametrize('method', eb_data['methods'])
def test_all_bounds_equal(method):

    def f(x, p1=1):
        return np.linalg.norm(x) + p1
    bounds = [(1, 1), (2, 2)]
    x0 = (1.0, 3.0)
    res = optimize.minimize(f, x0, bounds=bounds, method=method)
    assert res.success
    assert_allclose(res.fun, f([1.0, 2.0]))
    assert res.nfev == 1
    assert res.message == 'All independent variables were fixed by bounds.'
    args = (2,)
    res = optimize.minimize(f, x0, bounds=bounds, method=method, args=args)
    assert res.success
    assert_allclose(res.fun, f([1.0, 2.0], 2))
    if method.upper() == 'SLSQP':

        def con(x):
            return np.sum(x)
        nlc = NonlinearConstraint(con, -np.inf, 0.0)
        res = optimize.minimize(f, x0, bounds=bounds, method=method, constraints=[nlc])
        assert res.success is False
        assert_allclose(res.fun, f([1.0, 2.0]))
        assert res.nfev == 1
        message = 'All independent variables were fixed by bounds, but'
        assert res.message.startswith(message)
        nlc = NonlinearConstraint(con, -np.inf, 4)
        res = optimize.minimize(f, x0, bounds=bounds, method=method, constraints=[nlc])
        assert res.success is True
        assert_allclose(res.fun, f([1.0, 2.0]))
        assert res.nfev == 1
        message = 'All independent variables were fixed by bounds at values'
        assert res.message.startswith(message)