import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def test_dummy_regressor_return_std():
    X = [[0]] * 3
    y = np.array([2, 2, 2])
    y_std_expected = np.array([0, 0, 0])
    cls = DummyRegressor()
    cls.fit(X, y)
    y_pred_list = cls.predict(X, return_std=True)
    assert len(y_pred_list) == 2
    assert_array_equal(y_pred_list[1], y_std_expected)