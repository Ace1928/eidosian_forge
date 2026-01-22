import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def test_constant_strategy_regressor(global_random_seed):
    random_state = np.random.RandomState(seed=global_random_seed)
    X = [[0]] * 5
    y = random_state.randn(5)
    reg = DummyRegressor(strategy='constant', constant=[43])
    reg.fit(X, y)
    assert_array_equal(reg.predict(X), [43] * len(X))
    reg = DummyRegressor(strategy='constant', constant=43)
    reg.fit(X, y)
    assert_array_equal(reg.predict(X), [43] * len(X))
    assert not isinstance(reg.constant, np.ndarray)