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
def test_himmelblau(self):
    x0 = np.array(himmelblau_x0)
    sol = optimize.minimize(himmelblau, x0, jac=himmelblau_grad, hess=himmelblau_hess, method='Newton-CG', tol=1e-06)
    assert sol.success, sol.message
    assert_allclose(sol.x, himmelblau_xopt, rtol=0.0001)
    assert_allclose(sol.fun, himmelblau_min, atol=0.0001)