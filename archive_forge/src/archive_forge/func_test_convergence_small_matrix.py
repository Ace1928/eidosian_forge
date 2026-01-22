import pytest
import numpy as np
from numpy.linalg import lstsq
from numpy.testing import assert_allclose, assert_equal, assert_
from scipy.sparse import rand, coo_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import lsq_linear
from scipy.optimize._minimize import Bounds
def test_convergence_small_matrix(self):
    A = np.array([[49.0, 41.0, -32.0], [-19.0, -32.0, -8.0], [-13.0, 10.0, 69.0]])
    b = np.array([-41.0, -90.0, 47.0])
    bounds = np.array([[31.0, -44.0, 26.0], [54.0, -32.0, 28.0]])
    x_bvls = lsq_linear(A, b, bounds=bounds, method='bvls').x
    x_trf = lsq_linear(A, b, bounds=bounds, method='trf').x
    cost_bvls = np.sum((A @ x_bvls - b) ** 2)
    cost_trf = np.sum((A @ x_trf - b) ** 2)
    assert_(abs(cost_bvls - cost_trf) < cost_trf * 1e-10)