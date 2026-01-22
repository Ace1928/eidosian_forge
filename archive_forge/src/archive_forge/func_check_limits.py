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
def check_limits(self, method, default_iters):
    for start_v in [[0.1, 0.1], [1, 1], [2, 2]]:
        for mfev in [50, 500, 5000]:
            self.funcalls = 0
            res = optimize.minimize(self.slow_func, start_v, method=method, options={'maxfev': mfev})
            assert self.funcalls == res['nfev']
            if res['success']:
                assert res['nfev'] < mfev
            else:
                assert res['nfev'] >= mfev
        for mit in [50, 500, 5000]:
            res = optimize.minimize(self.slow_func, start_v, method=method, options={'maxiter': mit})
            if res['success']:
                assert res['nit'] <= mit
            else:
                assert res['nit'] >= mit
        for mfev, mit in [[50, 50], [5000, 5000], [5000, np.inf]]:
            self.funcalls = 0
            res = optimize.minimize(self.slow_func, start_v, method=method, options={'maxiter': mit, 'maxfev': mfev})
            assert self.funcalls == res['nfev']
            if res['success']:
                assert res['nfev'] < mfev and res['nit'] <= mit
            else:
                assert res['nfev'] >= mfev or res['nit'] >= mit
        for mfev, mit in [[np.inf, None], [None, np.inf]]:
            self.funcalls = 0
            res = optimize.minimize(self.slow_func, start_v, method=method, options={'maxiter': mit, 'maxfev': mfev})
            assert self.funcalls == res['nfev']
            if res['success']:
                if mfev is None:
                    assert res['nfev'] < default_iters * 2
                else:
                    assert res['nit'] <= default_iters * 2
            else:
                assert res['nfev'] >= default_iters * 2 or res['nit'] >= default_iters * 2