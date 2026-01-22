import numpy as np
import pytest
import scipy.sparse as sp
from numpy.random import RandomState
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import linalg
from sklearn.datasets import make_classification
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.sparsefuncs import (
from sklearn.utils.sparsefuncs_fast import (
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('sparse_constructor', CSC_CONTAINERS + CSR_CONTAINERS)
def test_mean_variance_axis0_precision(dtype, sparse_constructor):
    rng = np.random.RandomState(0)
    X = np.full(fill_value=100.0, shape=(1000, 1), dtype=dtype)
    missing_indices = rng.choice(np.arange(X.shape[0]), 10, replace=False)
    X[missing_indices, 0] = np.nan
    X = sparse_constructor(X)
    sample_weight = rng.rand(X.shape[0]).astype(dtype)
    _, var = mean_variance_axis(X, weights=sample_weight, axis=0)
    assert var < np.finfo(dtype).eps