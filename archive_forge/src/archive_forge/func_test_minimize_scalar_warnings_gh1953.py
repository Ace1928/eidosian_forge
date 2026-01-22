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
@pytest.mark.parametrize('method', ['bounded', 'brent', 'golden'])
def test_minimize_scalar_warnings_gh1953(self, method):

    def f(x):
        return (x - 1) ** 2
    kwargs = {}
    kwd = 'bounds' if method == 'bounded' else 'bracket'
    kwargs[kwd] = [-2, 10]
    options = {'disp': True, 'maxiter': 3}
    with pytest.warns(optimize.OptimizeWarning, match='Maximum number'):
        optimize.minimize_scalar(f, method=method, options=options, **kwargs)
    options['disp'] = False
    optimize.minimize_scalar(f, method=method, options=options, **kwargs)