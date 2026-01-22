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
@pytest.mark.parametrize('data, n_components, expected_solver', [(np.random.RandomState(0).uniform(size=(1000, 50)), 0.5, 'full'), (np.random.RandomState(0).uniform(size=(10, 50)), 5, 'full'), (np.random.RandomState(0).uniform(size=(1000, 50)), 50, 'full'), (np.random.RandomState(0).uniform(size=(1000, 50)), 10, 'randomized')])
def test_pca_svd_solver_auto(data, n_components, expected_solver):
    pca_auto = PCA(n_components=n_components, random_state=0)
    pca_test = PCA(n_components=n_components, svd_solver=expected_solver, random_state=0)
    pca_auto.fit(data)
    pca_test.fit(data)
    assert_allclose(pca_auto.components_, pca_test.components_)