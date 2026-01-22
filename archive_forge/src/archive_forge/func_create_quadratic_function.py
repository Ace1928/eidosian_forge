import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from scipy.optimize._constraints import (NonlinearConstraint, Bounds,
from scipy.optimize._trustregion_constr.canonical_constraint \
def create_quadratic_function(n, m, rng):
    a = rng.rand(m)
    A = rng.rand(m, n)
    H = rng.rand(m, n, n)
    HT = np.transpose(H, (1, 2, 0))

    def fun(x):
        return a + A.dot(x) + 0.5 * H.dot(x).dot(x)

    def jac(x):
        return A + H.dot(x)

    def hess(x, v):
        return HT.dot(v)
    return (fun, jac, hess)