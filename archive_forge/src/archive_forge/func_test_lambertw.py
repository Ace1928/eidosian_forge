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
def test_lambertw(self):
    xxroot = fixed_point(lambda xx: np.exp(-2.0 * xx) / 2.0, 1.0, args=(), xtol=1e-12, maxiter=500)
    assert_allclose(xxroot, np.exp(-2.0 * xxroot) / 2.0)
    assert_allclose(xxroot, lambertw(1) / 2)