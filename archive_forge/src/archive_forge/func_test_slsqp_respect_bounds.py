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
def test_slsqp_respect_bounds(self):

    def f(x):
        return sum((x - np.array([1.0, 2.0, 3.0, 4.0])) ** 2)

    def cons(x):
        a = np.array([[-1, -1, -1, -1], [-3, -3, -2, -1]])
        return np.concatenate([np.dot(a, x) + np.array([5, 10]), x])
    x0 = np.array([0.5, 1.0, 1.5, 2.0])
    res = optimize.minimize(f, x0, method='slsqp', constraints={'type': 'ineq', 'fun': cons})
    assert_allclose(res.x, np.array([0.0, 2, 5, 8]) / 3, atol=1e-12)