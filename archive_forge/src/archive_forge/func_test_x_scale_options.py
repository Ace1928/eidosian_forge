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
def test_x_scale_options(self):
    for x_scale in [1.0, np.array([0.5]), 'jac']:
        res = least_squares(fun_trivial, 2.0, x_scale=x_scale)
        assert_allclose(res.x, 0)
    assert_raises(ValueError, least_squares, fun_trivial, 2.0, x_scale='auto', method=self.method)
    assert_raises(ValueError, least_squares, fun_trivial, 2.0, x_scale=-1.0, method=self.method)
    assert_raises(ValueError, least_squares, fun_trivial, 2.0, x_scale=None, method=self.method)
    assert_raises(ValueError, least_squares, fun_trivial, 2.0, x_scale=1.0 + 2j, method=self.method)