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
def test_gh_19103():
    ydata = np.array([0.0] * 66 + [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 3.0, 1.0, 6.0, 5.0, 0.0, 0.0, 2.0, 8.0, 4.0, 4.0, 6.0, 9.0, 7.0, 2.0, 7.0, 8.0, 2.0, 13.0, 9.0, 8.0, 11.0, 10.0, 13.0, 14.0, 19.0, 11.0, 15.0, 18.0, 26.0, 19.0, 32.0, 29.0, 28.0, 36.0, 32.0, 35.0, 36.0, 43.0, 52.0, 32.0, 58.0, 56.0, 52.0, 67.0, 53.0, 72.0, 88.0, 77.0, 95.0, 94.0, 84.0, 86.0, 101.0, 107.0, 108.0, 118.0, 96.0, 115.0, 138.0, 137.0])
    xdata = np.arange(0, ydata.size) * 0.1

    def exponential_wrapped(params):
        A, B, x0 = params
        return A * np.exp(B * (xdata - x0)) - ydata
    x0 = [0.01, 1.0, 5.0]
    bounds = ((0.01, 0, 0), (np.inf, 10, 20.9))
    res = least_squares(exponential_wrapped, x0, method='trf', bounds=bounds)
    assert res.success