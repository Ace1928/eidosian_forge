import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
@pytest.mark.parametrize('strategy', ['stratified', 'most_frequent', 'prior', 'uniform', 'constant'])
def test_classifier_prediction_independent_of_X(strategy, global_random_seed):
    y = [0, 2, 1, 1]
    X1 = [[0]] * 4
    clf1 = DummyClassifier(strategy=strategy, random_state=global_random_seed, constant=0)
    clf1.fit(X1, y)
    predictions1 = clf1.predict(X1)
    X2 = [[1]] * 4
    clf2 = DummyClassifier(strategy=strategy, random_state=global_random_seed, constant=0)
    clf2.fit(X2, y)
    predictions2 = clf2.predict(X2)
    assert_array_equal(predictions1, predictions2)