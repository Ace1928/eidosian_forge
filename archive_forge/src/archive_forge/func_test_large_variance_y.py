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
def test_large_variance_y():
    """
    Here we test that, when noramlize_y=True, our GP can produce a
    sensible fit to training data whose variance is significantly
    larger than unity. This test was made in response to issue #15612.

    GP predictions are verified against predictions that were made
    using GPy which, here, is treated as the 'gold standard'. Note that we
    only investigate the RBF kernel here, as that is what was used in the
    GPy implementation.

    The following code can be used to recreate the GPy data:

    --------------------------------------------------------------------------
    import GPy

    kernel_gpy = GPy.kern.RBF(input_dim=1, lengthscale=1.)
    gpy = GPy.models.GPRegression(X, np.vstack(y_large), kernel_gpy)
    gpy.optimize()
    y_pred_gpy, y_var_gpy = gpy.predict(X2)
    y_pred_std_gpy = np.sqrt(y_var_gpy)
    --------------------------------------------------------------------------
    """
    y_large = 10 * y
    RBF_params = {'length_scale': 1.0}
    kernel = RBF(**RBF_params)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr.fit(X, y_large)
    y_pred, y_pred_std = gpr.predict(X2, return_std=True)
    y_pred_gpy = np.array([15.16918303, -27.98707845, -39.31636019, 14.52605515, 69.18503589])
    y_pred_std_gpy = np.array([7.78860962, 3.83179178, 0.63149951, 0.52745188, 0.86170042])
    assert_allclose(y_pred, y_pred_gpy, rtol=0.07, atol=0)
    assert_allclose(y_pred_std, y_pred_std_gpy, rtol=0.15, atol=0)