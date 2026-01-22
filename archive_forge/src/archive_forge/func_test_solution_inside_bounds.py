import re
import sys
import warnings
import numpy as np
import pytest
from scipy.optimize import approx_fprime
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
from sklearn.utils._testing import (
@pytest.mark.parametrize('kernel', non_fixed_kernels)
def test_solution_inside_bounds(kernel):
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    bounds = gpr.kernel_.bounds
    max_ = np.finfo(gpr.kernel_.theta.dtype).max
    tiny = 1e-10
    bounds[~np.isfinite(bounds[:, 1]), 1] = max_
    assert_array_less(bounds[:, 0], gpr.kernel_.theta + tiny)
    assert_array_less(gpr.kernel_.theta, bounds[:, 1] + tiny)