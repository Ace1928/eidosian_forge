import sys
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA, MiniBatchSparsePCA, SparsePCA
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_spca_early_stopping(global_random_seed):
    """Check that `tol` and `max_no_improvement` act as early stopping."""
    rng = np.random.RandomState(global_random_seed)
    n_samples, n_features = (50, 10)
    X = rng.randn(n_samples, n_features)
    model_early_stopped = MiniBatchSparsePCA(max_iter=100, tol=0.5, random_state=global_random_seed).fit(X)
    model_not_early_stopped = MiniBatchSparsePCA(max_iter=100, tol=0.001, random_state=global_random_seed).fit(X)
    assert model_early_stopped.n_iter_ < model_not_early_stopped.n_iter_
    model_early_stopped = MiniBatchSparsePCA(max_iter=100, tol=1e-06, max_no_improvement=2, random_state=global_random_seed).fit(X)
    model_not_early_stopped = MiniBatchSparsePCA(max_iter=100, tol=1e-06, max_no_improvement=100, random_state=global_random_seed).fit(X)
    assert model_early_stopped.n_iter_ < model_not_early_stopped.n_iter_