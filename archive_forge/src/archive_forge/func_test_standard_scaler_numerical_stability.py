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
@skip_if_32bit
def test_standard_scaler_numerical_stability():
    x = np.full(8, np.log(1e-05), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        scale(x)
    assert_array_almost_equal(scale(x), np.zeros(8))
    x = np.full(10, np.log(1e-05), dtype=np.float64)
    warning_message = 'standard deviation of the data is probably very close to 0'
    with pytest.warns(UserWarning, match=warning_message):
        x_scaled = scale(x)
    assert_array_almost_equal(x_scaled, np.zeros(10))
    x = np.full(10, 1e-100, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        x_small_scaled = scale(x)
    assert_array_almost_equal(x_small_scaled, np.zeros(10))
    x_big = np.full(10, 1e+100, dtype=np.float64)
    warning_message = 'Dataset may contain too large values'
    with pytest.warns(UserWarning, match=warning_message):
        x_big_scaled = scale(x_big)
    assert_array_almost_equal(x_big_scaled, np.zeros(10))
    assert_array_almost_equal(x_big_scaled, x_small_scaled)
    with pytest.warns(UserWarning, match=warning_message):
        x_big_centered = scale(x_big, with_std=False)
    assert_array_almost_equal(x_big_centered, np.zeros(10))
    assert_array_almost_equal(x_big_centered, x_small_scaled)