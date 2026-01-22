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
def test_pca_dim():
    rng = np.random.RandomState(0)
    n, p = (100, 5)
    X = rng.randn(n, p) * 0.1
    X[:10] += np.array([3, 4, 5, 1, 2])
    pca = PCA(n_components='mle', svd_solver='full').fit(X)
    assert pca.n_components == 'mle'
    assert pca.n_components_ == 1