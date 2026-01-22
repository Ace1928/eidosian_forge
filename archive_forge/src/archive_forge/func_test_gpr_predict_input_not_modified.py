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
def test_gpr_predict_input_not_modified():
    """
    Check that the input X is not modified by the predict method of the
    GaussianProcessRegressor when setting return_std=True.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/24340
    """
    gpr = GaussianProcessRegressor(kernel=CustomKernel()).fit(X, y)
    X2_copy = np.copy(X2)
    _, _ = gpr.predict(X2, return_std=True)
    assert_allclose(X2, X2_copy)