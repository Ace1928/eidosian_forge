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
def test_assess_dimesion_rank_one():
    n_samples, n_features = (9, 6)
    X = np.ones((n_samples, n_features))
    _, s, _ = np.linalg.svd(X, full_matrices=True)
    assert_allclose(s[1:], np.zeros(n_features - 1), atol=1e-12)
    assert np.isfinite(_assess_dimension(s, rank=1, n_samples=n_samples))
    for rank in range(2, n_features):
        assert _assess_dimension(s, rank, n_samples) == -np.inf