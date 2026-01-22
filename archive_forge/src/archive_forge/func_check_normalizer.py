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
def check_normalizer(norm, X_norm):
    """
    Convenient checking function for `test_normalizer_l1_l2_max` and
    `test_normalizer_l1_l2_max_non_csr`
    """
    if norm == 'l1':
        row_sums = np.abs(X_norm).sum(axis=1)
        for i in range(3):
            assert_almost_equal(row_sums[i], 1.0)
        assert_almost_equal(row_sums[3], 0.0)
    elif norm == 'l2':
        for i in range(3):
            assert_almost_equal(la.norm(X_norm[i]), 1.0)
        assert_almost_equal(la.norm(X_norm[3]), 0.0)
    elif norm == 'max':
        row_maxs = abs(X_norm).max(axis=1)
        for i in range(3):
            assert_almost_equal(row_maxs[i], 1.0)
        assert_almost_equal(row_maxs[3], 0.0)