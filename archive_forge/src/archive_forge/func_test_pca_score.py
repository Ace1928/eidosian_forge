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
@pytest.mark.parametrize('svd_solver', PCA_SOLVERS)
def test_pca_score(svd_solver):
    n, p = (1000, 3)
    rng = np.random.RandomState(0)
    X = rng.randn(n, p) * 0.1 + np.array([3, 4, 5])
    pca = PCA(n_components=2, svd_solver=svd_solver)
    pca.fit(X)
    ll1 = pca.score(X)
    h = -0.5 * np.log(2 * np.pi * np.exp(1) * 0.1 ** 2) * p
    assert_allclose(ll1 / h, 1, rtol=0.05)
    ll2 = pca.score(rng.randn(n, p) * 0.2 + np.array([3, 4, 5]))
    assert ll1 > ll2
    pca = PCA(n_components=2, whiten=True, svd_solver=svd_solver)
    pca.fit(X)
    ll2 = pca.score(X)
    assert ll1 > ll2