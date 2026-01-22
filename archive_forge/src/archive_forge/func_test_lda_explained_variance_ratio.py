import numpy as np
import pytest
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf, ShrunkCovariance, ledoit_wolf
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import (
from sklearn.preprocessing import StandardScaler
from sklearn.utils import _IS_WASM, check_random_state
from sklearn.utils._testing import (
def test_lda_explained_variance_ratio():
    state = np.random.RandomState(0)
    X = state.normal(loc=0, scale=100, size=(40, 20))
    y = state.randint(0, 3, size=(40,))
    clf_lda_eigen = LinearDiscriminantAnalysis(solver='eigen')
    clf_lda_eigen.fit(X, y)
    assert_almost_equal(clf_lda_eigen.explained_variance_ratio_.sum(), 1.0, 3)
    assert clf_lda_eigen.explained_variance_ratio_.shape == (2,), 'Unexpected length for explained_variance_ratio_'
    clf_lda_svd = LinearDiscriminantAnalysis(solver='svd')
    clf_lda_svd.fit(X, y)
    assert_almost_equal(clf_lda_svd.explained_variance_ratio_.sum(), 1.0, 3)
    assert clf_lda_svd.explained_variance_ratio_.shape == (2,), 'Unexpected length for explained_variance_ratio_'
    assert_array_almost_equal(clf_lda_svd.explained_variance_ratio_, clf_lda_eigen.explained_variance_ratio_)