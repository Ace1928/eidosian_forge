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
def test_powell_limits():
    bounds = optimize.Bounds([0, 0], [0.6, 20])

    def fun(x):
        a, b = x
        assert (x >= bounds.lb).all() and (x <= bounds.ub).all()
        return a ** 2 + b ** 2
    optimize.minimize(fun, x0=[0.6, 20], method='Powell', bounds=bounds)
    bounds = optimize.Bounds(lb=[0], ub=[1], keep_feasible=[True])

    def func(x):
        assert x >= 0 and x <= 1
        return np.exp(x)
    optimize.minimize(fun=func, x0=[0.5], method='powell', bounds=bounds)