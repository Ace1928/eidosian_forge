import re
import warnings
import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse, stats
from sklearn import datasets
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale
from sklearn.svm import SVR
from sklearn.utils import gen_batches, shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import (
from sklearn.utils.sparsefuncs import mean_variance_axis
@pytest.mark.parametrize('sparse_container', [None] + CSC_CONTAINERS + CSR_CONTAINERS)
@pytest.mark.parametrize('add_sample_weight', [False, True])
def test_standard_scaler_dtype(add_sample_weight, sparse_container):
    rng = np.random.RandomState(0)
    n_samples = 10
    n_features = 3
    if add_sample_weight:
        sample_weight = np.ones(n_samples)
    else:
        sample_weight = None
    with_mean = True
    for dtype in [np.float16, np.float32, np.float64]:
        X = rng.randn(n_samples, n_features).astype(dtype)
        if sparse_container is not None:
            X = sparse_container(X)
            with_mean = False
        scaler = StandardScaler(with_mean=with_mean)
        X_scaled = scaler.fit(X, sample_weight=sample_weight).transform(X)
        assert X.dtype == X_scaled.dtype
        assert scaler.mean_.dtype == np.float64
        assert scaler.scale_.dtype == np.float64