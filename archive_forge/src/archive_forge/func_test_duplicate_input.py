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
def test_duplicate_input(kernel):
    gpr_equal_inputs = GaussianProcessRegressor(kernel=kernel, alpha=0.01)
    gpr_similar_inputs = GaussianProcessRegressor(kernel=kernel, alpha=0.01)
    X_ = np.vstack((X, X[0]))
    y_ = np.hstack((y, y[0] + 1))
    gpr_equal_inputs.fit(X_, y_)
    X_ = np.vstack((X, X[0] + 1e-15))
    y_ = np.hstack((y, y[0] + 1))
    gpr_similar_inputs.fit(X_, y_)
    X_test = np.linspace(0, 10, 100)[:, None]
    y_pred_equal, y_std_equal = gpr_equal_inputs.predict(X_test, return_std=True)
    y_pred_similar, y_std_similar = gpr_similar_inputs.predict(X_test, return_std=True)
    assert_almost_equal(y_pred_equal, y_pred_similar)
    assert_almost_equal(y_std_equal, y_std_similar)