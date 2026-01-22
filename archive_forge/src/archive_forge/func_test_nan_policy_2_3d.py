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
@pytest.mark.parametrize('n', [2, 3])
@pytest.mark.parametrize('method', ['lm', 'trf', 'dogbox'])
def test_nan_policy_2_3d(self, n, method):

    def f(x, a, b):
        x1 = x[..., 0, :].squeeze()
        x2 = x[..., 1, :].squeeze()
        return a * x1 + b + x2
    xdata_with_nan = np.array([[[2, 3, np.nan, 4, 4, np.nan, 5], [2, 3, np.nan, np.nan, 4, np.nan, 7]]])
    xdata_with_nan = xdata_with_nan.squeeze() if n == 2 else xdata_with_nan
    ydata_with_nan = np.array([1, 2, 5, 3, np.nan, 7, 10])
    xdata_without_nan = np.array([[[2, 3, 5], [2, 3, 7]]])
    ydata_without_nan = np.array([1, 2, 10])
    self._check_nan_policy(f, xdata_with_nan, xdata_without_nan, ydata_with_nan, ydata_without_nan, method)