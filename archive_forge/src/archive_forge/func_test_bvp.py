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
def test_bvp(self):
    n = 10
    x0 = np.ones(n ** 2)
    if self.method == 'lm':
        max_nfev = 5000
    else:
        max_nfev = 100
    res = least_squares(fun_bvp, x0, ftol=0.01, method=self.method, max_nfev=max_nfev)
    assert_(res.nfev < max_nfev)
    assert_(res.cost < 0.5)