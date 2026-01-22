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
def test_no_empty_slice_warning():
    n_components = 10
    n_features = n_components + 2
    X = np.random.uniform(-1, 1, size=(n_components, n_features))
    pca = PCA(n_components=n_components)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        pca.fit(X)