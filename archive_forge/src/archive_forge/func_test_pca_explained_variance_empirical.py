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
@pytest.mark.parametrize('X', [np.random.RandomState(0).randn(100, 80), datasets.make_classification(100, 80, n_informative=78, random_state=0)[0]], ids=['random-data', 'correlated-data'])
@pytest.mark.parametrize('svd_solver', PCA_SOLVERS)
def test_pca_explained_variance_empirical(X, svd_solver):
    pca = PCA(n_components=2, svd_solver=svd_solver, random_state=0)
    X_pca = pca.fit_transform(X)
    assert_allclose(pca.explained_variance_, np.var(X_pca, ddof=1, axis=0))
    expected_result = np.linalg.eig(np.cov(X, rowvar=False))[0]
    expected_result = sorted(expected_result, reverse=True)[:2]
    assert_allclose(pca.explained_variance_, expected_result, rtol=0.005)