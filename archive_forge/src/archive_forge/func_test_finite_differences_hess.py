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
def test_finite_differences_hess(self):
    methods = ['trust-constr', 'Newton-CG', 'trust-ncg', 'trust-krylov']
    hesses = FD_METHODS + (optimize.BFGS,)
    for method, hess in itertools.product(methods, hesses):
        if hess is optimize.BFGS:
            hess = hess()
        result = optimize.minimize(self.func, self.startparams, method=method, jac=self.grad, hess=hess)
        assert result.success
    methods = ['trust-ncg', 'trust-krylov', 'dogleg', 'trust-exact']
    for method in methods:
        with pytest.raises(ValueError):
            optimize.minimize(self.func, self.startparams, method=method, jac=self.grad, hess=None)