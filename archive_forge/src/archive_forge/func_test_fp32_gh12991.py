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
def test_fp32_gh12991():
    np.random.seed(1)
    x = np.linspace(0, 1, 100).astype('float32')
    y = np.random.random(100).astype('float32')

    def func(p, x):
        return p[0] + p[1] * x

    def err(p, x, y):
        return func(p, x) - y
    res = least_squares(err, [-1.0, -1.0], args=(x, y))
    assert res.nfev > 2
    assert_allclose(res.x, np.array([0.4082241, 0.15530563]), atol=5e-05)