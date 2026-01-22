import re
import sys
import warnings
from io import StringIO
import numpy as np
import pytest
from scipy import linalg
from sklearn.base import clone
from sklearn.decomposition import NMF, MiniBatchNMF, non_negative_factorization
from sklearn.decomposition import _nmf as nmf  # For testing internals
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import (
from sklearn.utils.extmath import squared_norm
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_nmf_negative_beta_loss(csr_container):
    n_samples = 6
    n_features = 5
    n_components = 3
    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(n_samples, n_features)
    np.clip(X, 0, None, out=X)
    X_csr = csr_container(X)

    def _assert_nmf_no_nan(X, beta_loss):
        W, H, _ = non_negative_factorization(X, init='random', n_components=n_components, solver='mu', beta_loss=beta_loss, random_state=0, max_iter=1000)
        assert not np.any(np.isnan(W))
        assert not np.any(np.isnan(H))
    msg = 'When beta_loss <= 0 and X contains zeros, the solver may diverge.'
    for beta_loss in (-0.6, 0.0):
        with pytest.raises(ValueError, match=msg):
            _assert_nmf_no_nan(X, beta_loss)
        _assert_nmf_no_nan(X + 1e-09, beta_loss)
    for beta_loss in (0.2, 1.0, 1.2, 2.0, 2.5):
        _assert_nmf_no_nan(X, beta_loss)
        _assert_nmf_no_nan(X_csr, beta_loss)