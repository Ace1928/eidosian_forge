import re
import warnings
import numpy as np
import pytest
import scipy as sp
from numpy.testing import assert_array_equal
from sklearn import config_context, datasets
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.decomposition import PCA
from sklearn.decomposition._pca import _assess_dimension, _infer_dimension
from sklearn.utils._array_api import (
from sklearn.utils._array_api import device as array_device
from sklearn.utils._testing import _array_api_for_tests, assert_allclose
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_pca_randomized_svd_n_oversamples():
    """Check that exposing and setting `n_oversamples` will provide accurate results
    even when `X` as a large number of features.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/20589
    """
    rng = np.random.RandomState(0)
    n_features = 100
    X = rng.randn(1000, n_features)
    pca_randomized = PCA(n_components=1, svd_solver='randomized', n_oversamples=n_features, random_state=0).fit(X)
    pca_full = PCA(n_components=1, svd_solver='full').fit(X)
    pca_arpack = PCA(n_components=1, svd_solver='arpack', random_state=0).fit(X)
    assert_allclose(np.abs(pca_full.components_), np.abs(pca_arpack.components_))
    assert_allclose(np.abs(pca_randomized.components_), np.abs(pca_arpack.components_))