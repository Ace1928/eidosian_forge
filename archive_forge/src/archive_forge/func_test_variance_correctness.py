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
@pytest.mark.parametrize('copy', [True, False])
def test_variance_correctness(copy):
    """Check the accuracy of PCA's internal variance calculation"""
    rng = np.random.RandomState(0)
    X = rng.randn(1000, 200)
    pca = PCA().fit(X)
    pca_var = pca.explained_variance_ / pca.explained_variance_ratio_
    true_var = np.var(X, ddof=1, axis=0).sum()
    np.testing.assert_allclose(pca_var, true_var)