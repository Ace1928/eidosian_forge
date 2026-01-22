import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
def test_incremental_pca():
    X = iris.data
    batch_size = X.shape[0] // 3
    ipca = IncrementalPCA(n_components=2, batch_size=batch_size)
    pca = PCA(n_components=2)
    pca.fit_transform(X)
    X_transformed = ipca.fit_transform(X)
    assert X_transformed.shape == (X.shape[0], 2)
    np.testing.assert_allclose(ipca.explained_variance_ratio_.sum(), pca.explained_variance_ratio_.sum(), rtol=0.001)
    for n_components in [1, 2, X.shape[1]]:
        ipca = IncrementalPCA(n_components, batch_size=batch_size)
        ipca.fit(X)
        cov = ipca.get_covariance()
        precision = ipca.get_precision()
        np.testing.assert_allclose(np.dot(cov, precision), np.eye(X.shape[1]), atol=1e-13)