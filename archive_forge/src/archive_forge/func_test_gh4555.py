import warnings
import pytest
from numpy.testing import (assert_, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
import numpy as np
from numpy import array, float64
from multiprocessing.pool import ThreadPool
from scipy import optimize, linalg
from scipy.special import lambertw
from scipy.optimize._minpack_py import leastsq, curve_fit, fixed_point
from scipy.optimize import OptimizeWarning
from scipy.optimize._minimize import Bounds
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_gh4555(self):

    def f(x, a, b, c, d, e):
        return a * np.log(x + 1 + b) + c * np.log(x + 1 + d) + e
    rng = np.random.default_rng(408113519974467917)
    n = 100
    x = np.arange(n)
    y = np.linspace(2, 7, n) + rng.random(n)
    p, cov = optimize.curve_fit(f, x, y, maxfev=100000)
    assert np.all(np.diag(cov) > 0)
    eigs = linalg.eigh(cov)[0]
    assert np.all(eigs > -0.01)
    assert_allclose(cov, cov.T)