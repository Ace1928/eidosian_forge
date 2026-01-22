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
def test_pca_deterministic_output(svd_solver):
    rng = np.random.RandomState(0)
    X = rng.rand(10, 10)
    transformed_X = np.zeros((20, 2))
    for i in range(20):
        pca = PCA(n_components=2, svd_solver=svd_solver, random_state=rng)
        transformed_X[i, :] = pca.fit_transform(X)[0]
    assert_allclose(transformed_X, np.tile(transformed_X[0, :], 20).reshape(20, 2))