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
def test_solver_selection(self):
    sparse = BroydenTridiagonal(mode='sparse')
    dense = BroydenTridiagonal(mode='dense')
    res_sparse = least_squares(sparse.fun, sparse.x0, jac=sparse.jac, method=self.method)
    res_dense = least_squares(dense.fun, dense.x0, jac=dense.jac, method=self.method)
    assert_allclose(res_sparse.cost, 0, atol=1e-20)
    assert_allclose(res_dense.cost, 0, atol=1e-20)
    assert_(issparse(res_sparse.jac))
    assert_(isinstance(res_dense.jac, np.ndarray))