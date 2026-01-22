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
@pytest.mark.parametrize('n_targets', [None, 1, 2, 3])
def test_predict_shape_with_prior(n_targets):
    """Check the output shape of `predict` with prior distribution."""
    rng = np.random.RandomState(1024)
    n_sample = 10
    X = rng.randn(n_sample, 3)
    y = rng.randn(n_sample, n_targets if n_targets is not None else 1)
    model = GaussianProcessRegressor(n_targets=n_targets)
    mean_prior, cov_prior = model.predict(X, return_cov=True)
    _, std_prior = model.predict(X, return_std=True)
    model.fit(X, y)
    mean_post, cov_post = model.predict(X, return_cov=True)
    _, std_post = model.predict(X, return_std=True)
    assert mean_prior.shape == mean_post.shape
    assert cov_prior.shape == cov_post.shape
    assert std_prior.shape == std_post.shape