import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_less
def test_truncated_svd_eq_pca(X_sparse):
    X_dense = X_sparse.toarray()
    X_c = X_dense - X_dense.mean(axis=0)
    params = dict(n_components=10, random_state=42)
    svd = TruncatedSVD(algorithm='arpack', **params)
    pca = PCA(svd_solver='arpack', **params)
    Xt_svd = svd.fit_transform(X_c)
    Xt_pca = pca.fit_transform(X_c)
    assert_allclose(Xt_svd, Xt_pca, rtol=1e-09)
    assert_allclose(pca.mean_, 0, atol=1e-09)
    assert_allclose(svd.components_, pca.components_)