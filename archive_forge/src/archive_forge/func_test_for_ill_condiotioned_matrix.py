import numpy as np
from scipy.optimize._trustregion_exact import (
from scipy.linalg import (svd, get_lapack_funcs, det, qr, norm)
from numpy.testing import (assert_array_equal,
def test_for_ill_condiotioned_matrix(self):
    C = np.array([[1, 2, 3, 4], [0, 0.05, 60, 7], [0, 0, 0.8, 9], [0, 0, 0, 10]])
    U, s, Vt = svd(C)
    smin_svd = s[-1]
    zmin_svd = Vt[-1, :]
    smin, zmin = estimate_smallest_singular_value(C)
    assert_array_almost_equal(smin, smin_svd, decimal=8)
    assert_array_almost_equal(abs(zmin), abs(zmin_svd), decimal=8)