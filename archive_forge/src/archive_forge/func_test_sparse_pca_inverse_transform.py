import sys
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA, MiniBatchSparsePCA, SparsePCA
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_sparse_pca_inverse_transform():
    """Check that `inverse_transform` in `SparsePCA` and `PCA` are similar."""
    rng = np.random.RandomState(0)
    n_samples, n_features = (10, 5)
    X = rng.randn(n_samples, n_features)
    n_components = 2
    spca = SparsePCA(n_components=n_components, alpha=1e-12, ridge_alpha=1e-12, random_state=0)
    pca = PCA(n_components=n_components, random_state=0)
    X_trans_spca = spca.fit_transform(X)
    X_trans_pca = pca.fit_transform(X)
    assert_allclose(spca.inverse_transform(X_trans_spca), pca.inverse_transform(X_trans_pca))