import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_less
@pytest.mark.parametrize('solver', SVD_SOLVERS)
def test_singular_values_expected(solver):
    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 110
    X = rng.randn(n_samples, n_features)
    pca = TruncatedSVD(n_components=3, algorithm=solver, random_state=rng)
    X_pca = pca.fit_transform(X)
    X_pca /= np.sqrt(np.sum(X_pca ** 2.0, axis=0))
    X_pca[:, 0] *= 3.142
    X_pca[:, 1] *= 2.718
    X_hat_pca = np.dot(X_pca, pca.components_)
    pca.fit(X_hat_pca)
    assert_allclose(pca.singular_values_, [3.142, 2.718, 1.0], rtol=1e-14)