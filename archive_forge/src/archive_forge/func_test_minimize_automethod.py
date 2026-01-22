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
def test_minimize_automethod(self):

    def f(x):
        return x ** 2

    def cons(x):
        return x - 2
    x0 = np.array([10.0])
    sol_0 = optimize.minimize(f, x0)
    sol_1 = optimize.minimize(f, x0, constraints=[{'type': 'ineq', 'fun': cons}])
    sol_2 = optimize.minimize(f, x0, bounds=[(5, 10)])
    sol_3 = optimize.minimize(f, x0, constraints=[{'type': 'ineq', 'fun': cons}], bounds=[(5, 10)])
    sol_4 = optimize.minimize(f, x0, constraints=[{'type': 'ineq', 'fun': cons}], bounds=[(1, 10)])
    for sol in [sol_0, sol_1, sol_2, sol_3, sol_4]:
        assert sol.success
    assert_allclose(sol_0.x, 0, atol=1e-07)
    assert_allclose(sol_1.x, 2, atol=1e-07)
    assert_allclose(sol_2.x, 5, atol=1e-07)
    assert_allclose(sol_3.x, 5, atol=1e-07)
    assert_allclose(sol_4.x, 2, atol=1e-07)