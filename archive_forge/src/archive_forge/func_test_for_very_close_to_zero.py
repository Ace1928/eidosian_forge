import numpy as np
from scipy.optimize._trlib import (get_trlib_quadratic_subproblem)
from numpy.testing import (assert_,
def test_for_very_close_to_zero(self):
    H = np.array([[0.88547534, 2.90692271, 0.98440885, -0.78911503, -0.28035809], [2.90692271, -0.04618819, 0.32867263, -0.83737945, 0.17116396], [0.98440885, 0.32867263, -0.87355957, -0.06521957, -1.43030957], [-0.78911503, -0.83737945, -0.06521957, -1.645709, -0.33887298], [-0.28035809, 0.17116396, -1.43030957, -0.33887298, -1.68586978]])
    g = np.array([0, 0, 0, 0, 1e-06])
    trust_radius = 1.1
    subprob = KrylovQP(x=0, fun=lambda x: 0, jac=lambda x: g, hess=lambda x: None, hessp=lambda x, y: H.dot(y))
    p, hits_boundary = subprob.solve(trust_radius)
    assert_almost_equal(np.linalg.norm(H.dot(p) + subprob.lam * p + g), 0.0)
    assert_almost_equal(np.linalg.norm(p), trust_radius)
    assert_array_almost_equal(p, [0.06910534, -0.01432721, -0.65311947, -0.23815972, -0.84954934])
    assert_array_almost_equal(hits_boundary, True)