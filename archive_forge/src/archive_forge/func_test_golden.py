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
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_golden(self):
    x = optimize.golden(self.fun)
    assert_allclose(x, self.solution, atol=1e-06)
    x = optimize.golden(self.fun, brack=(-3, -2))
    assert_allclose(x, self.solution, atol=1e-06)
    x = optimize.golden(self.fun, full_output=True)
    assert_allclose(x[0], self.solution, atol=1e-06)
    x = optimize.golden(self.fun, brack=(-15, -1, 15))
    assert_allclose(x, self.solution, atol=1e-06)
    x = optimize.golden(self.fun, tol=0)
    assert_allclose(x, self.solution)
    maxiter_test_cases = [0, 1, 5]
    for maxiter in maxiter_test_cases:
        x0 = optimize.golden(self.fun, maxiter=0, full_output=True)
        x = optimize.golden(self.fun, maxiter=maxiter, full_output=True)
        nfev0, nfev = (x0[2], x[2])
        assert_equal(nfev - nfev0, maxiter)
    message = '\\(f\\(xb\\) < f\\(xa\\)\\) and \\(f\\(xb\\) < f\\(xc\\)\\)'
    with pytest.raises(ValueError, match=message):
        optimize.golden(self.fun, brack=(-1, 0, 1))
    message = '\\(xa < xb\\) and \\(xb < xc\\)'
    with pytest.raises(ValueError, match=message):
        optimize.golden(self.fun, brack=(0, -1, 1))