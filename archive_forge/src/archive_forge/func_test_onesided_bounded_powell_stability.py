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
def test_onesided_bounded_powell_stability():
    kwargs = {'method': 'Powell', 'bounds': [(-np.inf, 1000000.0)] * 3, 'options': {'ftol': 1e-08, 'xtol': 1e-08}}
    x0 = [1, 1, 1]

    def f(x):
        return -np.sum(x)
    res = optimize.minimize(f, x0, **kwargs)
    assert_allclose(res.fun, -3000000.0, atol=0.0001)

    def f(x):
        return -np.abs(np.sum(x)) ** 0.1 * (1 if np.all(x > 0) else -1)
    res = optimize.minimize(f, x0, **kwargs)
    assert_allclose(res.fun, -3000000.0 ** 0.1)

    def f(x):
        return -np.abs(np.sum(x)) ** 10 * (1 if np.all(x > 0) else -1)
    res = optimize.minimize(f, x0, **kwargs)
    assert_allclose(res.fun, -3000000.0 ** 10, rtol=1e-07)

    def f(x):
        t = -np.abs(np.sum(x[:2])) ** 5 - np.abs(np.sum(x[2:])) ** 0.1
        t *= 1 if np.all(x > 0) else -1
        return t
    kwargs['bounds'] = [(-np.inf, 1000.0)] * 3
    res = optimize.minimize(f, x0, **kwargs)
    assert_allclose(res.fun, -2000.0 ** 5 - 1000000.0 ** 0.1, rtol=1e-07)