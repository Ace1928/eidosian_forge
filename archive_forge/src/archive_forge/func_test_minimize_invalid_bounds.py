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
@pytest.mark.parametrize('method', ('nelder-mead', 'l-bfgs-b', 'tnc', 'powell', 'cobyla', 'trust-constr'))
def test_minimize_invalid_bounds(self, method):

    def f(x):
        return np.sum(x ** 2)
    bounds = Bounds([1, 2], [3, 4])
    msg = 'The number of bounds is not compatible with the length of `x0`.'
    with pytest.raises(ValueError, match=msg):
        optimize.minimize(f, x0=[1, 2, 3], method=method, bounds=bounds)
    bounds = Bounds([1, 6, 1], [3, 4, 2])
    msg = 'An upper bound is less than the corresponding lower bound.'
    with pytest.raises(ValueError, match=msg):
        optimize.minimize(f, x0=[1, 2, 3], method=method, bounds=bounds)