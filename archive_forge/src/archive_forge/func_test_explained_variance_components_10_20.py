import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_less
@pytest.mark.parametrize('kind', ('dense', 'sparse'))
@pytest.mark.parametrize('solver', SVD_SOLVERS)
def test_explained_variance_components_10_20(X_sparse, kind, solver):
    X = X_sparse if kind == 'sparse' else X_sparse.toarray()
    svd_10 = TruncatedSVD(10, algorithm=solver, n_iter=10).fit(X)
    svd_20 = TruncatedSVD(20, algorithm=solver, n_iter=10).fit(X)
    assert_allclose(svd_10.explained_variance_ratio_, svd_20.explained_variance_ratio_[:10], rtol=0.005)
    assert svd_20.explained_variance_ratio_.sum() > svd_10.explained_variance_ratio_.sum()