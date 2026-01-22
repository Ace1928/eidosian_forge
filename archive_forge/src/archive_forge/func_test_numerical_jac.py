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
def test_numerical_jac(self):
    p = BroydenTridiagonal()
    for jac in ['2-point', '3-point', 'cs']:
        res_dense = least_squares(p.fun, p.x0, jac, method=self.method)
        res_sparse = least_squares(p.fun, p.x0, jac, method=self.method, jac_sparsity=p.sparsity)
        assert_equal(res_dense.nfev, res_sparse.nfev)
        assert_allclose(res_dense.x, res_sparse.x, atol=1e-20)
        assert_allclose(res_dense.cost, 0, atol=1e-20)
        assert_allclose(res_sparse.cost, 0, atol=1e-20)