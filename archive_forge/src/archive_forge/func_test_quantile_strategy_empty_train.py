import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def test_quantile_strategy_empty_train():
    est = DummyRegressor(strategy='quantile', quantile=0.4)
    with pytest.raises(ValueError):
        est.fit([], [])