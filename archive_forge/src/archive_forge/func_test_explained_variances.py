import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
def test_explained_variances():
    X = datasets.make_low_rank_matrix(1000, 100, tail_strength=0.0, effective_rank=10, random_state=1999)
    prec = 3
    n_samples, n_features = X.shape
    for nc in [None, 99]:
        pca = PCA(n_components=nc).fit(X)
        ipca = IncrementalPCA(n_components=nc, batch_size=100).fit(X)
        assert_almost_equal(pca.explained_variance_, ipca.explained_variance_, decimal=prec)
        assert_almost_equal(pca.explained_variance_ratio_, ipca.explained_variance_ratio_, decimal=prec)
        assert_almost_equal(pca.noise_variance_, ipca.noise_variance_, decimal=prec)