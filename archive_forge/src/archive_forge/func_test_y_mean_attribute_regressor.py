import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def test_y_mean_attribute_regressor():
    X = [[0]] * 5
    y = [1, 2, 4, 6, 8]
    est = DummyRegressor(strategy='mean')
    est.fit(X, y)
    assert est.constant_ == np.mean(y)