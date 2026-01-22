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
def test_check_grad():

    def expit(x):
        return 1 / (1 + np.exp(-x))

    def der_expit(x):
        return np.exp(-x) / (1 + np.exp(-x)) ** 2
    x0 = np.array([1.5])
    r = optimize.check_grad(expit, der_expit, x0)
    assert_almost_equal(r, 0)
    r = optimize.check_grad(expit, der_expit, x0, direction='random', seed=1234)
    assert_almost_equal(r, 0)
    r = optimize.check_grad(expit, der_expit, x0, epsilon=1e-06)
    assert_almost_equal(r, 0)
    r = optimize.check_grad(expit, der_expit, x0, epsilon=1e-06, direction='random', seed=1234)
    assert_almost_equal(r, 0)
    r = abs(optimize.check_grad(expit, der_expit, x0, epsilon=0.1) - 0)
    assert r > 1e-07
    r = abs(optimize.check_grad(expit, der_expit, x0, epsilon=0.1, direction='random', seed=1234) - 0)
    assert r > 1e-07

    def x_sinx(x):
        return (x * np.sin(x)).sum()

    def der_x_sinx(x):
        return np.sin(x) + x * np.cos(x)
    x0 = np.arange(0, 2, 0.2)
    r = optimize.check_grad(x_sinx, der_x_sinx, x0, direction='random', seed=1234)
    assert_almost_equal(r, 0)
    assert_raises(ValueError, optimize.check_grad, x_sinx, der_x_sinx, x0, direction='random_projection', seed=1234)
    r = optimize.check_grad(himmelblau_grad, himmelblau_hess, himmelblau_x0, direction='all', seed=1234)
    assert r < 5e-07