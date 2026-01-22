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
@pytest.mark.parametrize('method', ('brent', 'golden'))
def test_minimize_scalar_success_false(self, method):

    def f(x):
        return x ** 2 if (-1 < x) & (x < 1) else 100.0
    message = 'The algorithm terminated without finding a valid bracket.'
    res = optimize.minimize_scalar(f, bracket=(-1, 1), method=method)
    assert not res.success
    assert message in res.message
    assert res.nfev == 3
    assert res.nit == 0
    assert res.fun == 100