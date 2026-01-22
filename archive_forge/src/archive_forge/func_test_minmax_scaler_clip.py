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
@pytest.mark.parametrize('feature_range', [(0, 1), (-10, 10)])
def test_minmax_scaler_clip(feature_range):
    X = iris.data
    scaler = MinMaxScaler(feature_range=feature_range, clip=True).fit(X)
    X_min, X_max = (np.min(X, axis=0), np.max(X, axis=0))
    X_test = [np.r_[X_min[:2] - 10, X_max[2:] + 10]]
    X_transformed = scaler.transform(X_test)
    assert_allclose(X_transformed, [[feature_range[0], feature_range[0], feature_range[1], feature_range[1]]])