import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
def test_n_components_none():
    rng = np.random.RandomState(1999)
    for n_samples, n_features in [(50, 10), (10, 50)]:
        X = rng.rand(n_samples, n_features)
        ipca = IncrementalPCA(n_components=None)
        ipca.partial_fit(X)
        assert ipca.n_components_ == min(X.shape)
        ipca.partial_fit(X)
        assert ipca.n_components_ == ipca.components_.shape[0]