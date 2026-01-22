import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
def test_incremental_pca_set_params():
    rng = np.random.RandomState(1999)
    n_samples = 100
    n_features = 20
    X = rng.randn(n_samples, n_features)
    X2 = rng.randn(n_samples, n_features)
    X3 = rng.randn(n_samples, n_features)
    ipca = IncrementalPCA(n_components=20)
    ipca.fit(X)
    ipca.set_params(n_components=10)
    with pytest.raises(ValueError):
        ipca.partial_fit(X2)
    ipca.set_params(n_components=15)
    with pytest.raises(ValueError):
        ipca.partial_fit(X3)
    ipca.set_params(n_components=20)
    ipca.partial_fit(X)