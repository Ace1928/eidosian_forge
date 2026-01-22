import numpy as np
from scipy.optimize._trustregion_exact import (
from scipy.linalg import (svd, get_lapack_funcs, det, qr, norm)
from numpy.testing import (assert_array_equal,
def test_for_jac_very_close_to_zero(self):
    H = [[0.88547534, 2.90692271, 0.98440885, -0.78911503, -0.28035809], [2.90692271, -0.04618819, 0.32867263, -0.83737945, 0.17116396], [0.98440885, 0.32867263, -0.87355957, -0.06521957, -1.43030957], [-0.78911503, -0.83737945, -0.06521957, -1.645709, -0.33887298], [-0.28035809, 0.17116396, -1.43030957, -0.33887298, -1.68586978]]
    g = [0, 0, 0, 0, 1e-15]
    subprob = IterativeSubproblem(x=0, fun=lambda x: 0, jac=lambda x: np.array(g), hess=lambda x: np.array(H), k_easy=1e-10, k_hard=1e-10)
    p, hits_boundary = subprob.solve(1.1)
    assert_array_almost_equal(p, [0.06910534, -0.01432721, -0.65311947, -0.23815972, -0.84954934])
    assert_array_almost_equal(hits_boundary, True)