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
def test_gh_18793_and_19351():
    answer = 1e-12
    initial_guess = 1.1e-12

    def chi2(x):
        return (x - answer) ** 2
    gtol = 1e-15
    res = least_squares(chi2, x0=initial_guess, gtol=1e-15, bounds=(0, np.inf))
    scaling, _ = CL_scaling_vector(res.x, res.grad, np.atleast_1d(0), np.atleast_1d(np.inf))
    assert res.status == 1
    assert np.linalg.norm(res.grad * scaling, ord=np.inf) < gtol