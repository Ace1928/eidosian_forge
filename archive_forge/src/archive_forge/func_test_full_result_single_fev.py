from itertools import product
import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import issparse, lil_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import least_squares, Bounds
from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES
from scipy.optimize._lsq.common import EPS, make_strictly_feasible, CL_scaling_vector
def test_full_result_single_fev(self):
    if self.method == 'lm':
        return
    res = least_squares(fun_trivial, 2.0, method=self.method, max_nfev=1)
    assert_equal(res.x, np.array([2]))
    assert_equal(res.cost, 40.5)
    assert_equal(res.fun, np.array([9]))
    assert_equal(res.jac, np.array([[4]]))
    assert_equal(res.grad, np.array([36]))
    assert_equal(res.optimality, 36)
    assert_equal(res.active_mask, np.array([0]))
    assert_equal(res.nfev, 1)
    assert_equal(res.njev, 1)
    assert_equal(res.status, 0)
    assert_equal(res.success, 0)