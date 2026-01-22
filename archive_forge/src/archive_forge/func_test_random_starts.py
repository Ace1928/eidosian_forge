import warnings
import numpy as np
import pytest
from scipy.optimize import approx_fprime
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
from sklearn.utils._testing import assert_almost_equal, assert_array_equal
def test_random_starts(global_random_seed):
    n_samples, n_features = (25, 2)
    rng = np.random.RandomState(global_random_seed)
    X = rng.randn(n_samples, n_features) * 2 - 1
    y = np.sin(X).sum(axis=1) + np.sin(3 * X).sum(axis=1) > 0
    kernel = C(1.0, (0.01, 100.0)) * RBF(length_scale=[0.001] * n_features, length_scale_bounds=[(0.0001, 100.0)] * n_features)
    last_lml = -np.inf
    for n_restarts_optimizer in range(5):
        gp = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, random_state=global_random_seed).fit(X, y)
        lml = gp.log_marginal_likelihood(gp.kernel_.theta)
        assert lml > last_lml - np.finfo(np.float32).eps
        last_lml = lml