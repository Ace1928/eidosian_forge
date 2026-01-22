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
def test_linear_operator(self):
    p = BroydenTridiagonal(mode='operator')
    res = least_squares(p.fun, p.x0, p.jac, method=self.method)
    assert_allclose(res.cost, 0.0, atol=1e-20)
    assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac, method=self.method, tr_solver='exact')