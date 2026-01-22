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
def test_standard_scaler_partial_fit():
    X = X_2d
    n = X.shape[0]
    for chunk_size in [1, 2, 50, n, n + 42]:
        scaler_batch = StandardScaler(with_std=False).fit(X)
        scaler_incr = StandardScaler(with_std=False)
        for batch in gen_batches(n_samples, chunk_size):
            scaler_incr = scaler_incr.partial_fit(X[batch])
        assert_array_almost_equal(scaler_batch.mean_, scaler_incr.mean_)
        assert scaler_batch.var_ == scaler_incr.var_
        assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_
        batch0 = slice(0, chunk_size)
        scaler_incr = StandardScaler().partial_fit(X[batch0])
        if chunk_size == 1:
            assert_array_almost_equal(np.zeros(n_features, dtype=np.float64), scaler_incr.var_)
            assert_array_almost_equal(np.ones(n_features, dtype=np.float64), scaler_incr.scale_)
        else:
            assert_array_almost_equal(np.var(X[batch0], axis=0), scaler_incr.var_)
            assert_array_almost_equal(np.std(X[batch0], axis=0), scaler_incr.scale_)
        scaler_batch = StandardScaler().fit(X)
        scaler_incr = StandardScaler()
        for i, batch in enumerate(gen_batches(n_samples, chunk_size)):
            scaler_incr = scaler_incr.partial_fit(X[batch])
            assert_correct_incr(i, batch_start=batch.start, batch_stop=batch.stop, n=n, chunk_size=chunk_size, n_samples_seen=scaler_incr.n_samples_seen_)
        assert_array_almost_equal(scaler_batch.var_, scaler_incr.var_)
        assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_