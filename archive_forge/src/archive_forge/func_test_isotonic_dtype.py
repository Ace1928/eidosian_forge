import copy
import pickle
import warnings
import numpy as np
import pytest
from scipy.special import expit
import sklearn
from sklearn.datasets import make_regression
from sklearn.isotonic import (
from sklearn.utils import shuffle
from sklearn.utils._testing import (
from sklearn.utils.validation import check_array
def test_isotonic_dtype():
    y = [2, 1, 4, 3, 5]
    weights = np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float64)
    reg = IsotonicRegression()
    for dtype in (np.int32, np.int64, np.float32, np.float64):
        for sample_weight in (None, weights.astype(np.float32), weights):
            y_np = np.array(y, dtype=dtype)
            expected_dtype = check_array(y_np, dtype=[np.float64, np.float32], ensure_2d=False).dtype
            res = isotonic_regression(y_np, sample_weight=sample_weight)
            assert res.dtype == expected_dtype
            X = np.arange(len(y)).astype(dtype)
            reg.fit(X, y_np, sample_weight=sample_weight)
            res = reg.predict(X)
            assert res.dtype == expected_dtype