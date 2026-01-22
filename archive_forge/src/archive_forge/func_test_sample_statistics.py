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
@pytest.mark.parametrize('kernel', kernels)
def test_sample_statistics(kernel):
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    y_mean, y_cov = gpr.predict(X2, return_cov=True)
    samples = gpr.sample_y(X2, 300000)
    assert_almost_equal(y_mean, np.mean(samples, 1), 1)
    assert_almost_equal(np.diag(y_cov) / np.diag(y_cov).max(), np.var(samples, 1) / np.diag(y_cov).max(), 1)