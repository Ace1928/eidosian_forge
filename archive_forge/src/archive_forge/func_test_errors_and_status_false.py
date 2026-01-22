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
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_errors_and_status_false(self):

    def f(x):
        return x ** 2 if (-1 < x) & (x < 1) else 100.0
    message = 'The algorithm terminated without finding a valid bracket.'
    with pytest.raises(RuntimeError, match=message):
        optimize.bracket(f, -1, 1)
    with pytest.raises(RuntimeError, match=message):
        optimize.bracket(f, -1, np.inf)
    with pytest.raises(RuntimeError, match=message):
        optimize.brent(f, brack=(-1, 1))
    with pytest.raises(RuntimeError, match=message):
        optimize.golden(f, brack=(-1, 1))

    def f(x):
        return -5 * x ** 5 + 4 * x ** 4 - 12 * x ** 3 + 11 * x ** 2 - 2 * x + 1
    message = 'No valid bracket was found before the iteration limit...'
    with pytest.raises(RuntimeError, match=message):
        optimize.bracket(f, -0.5, 0.5, maxiter=10)