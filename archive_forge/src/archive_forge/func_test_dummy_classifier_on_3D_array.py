import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def test_dummy_classifier_on_3D_array():
    X = np.array([[['foo']], [['bar']], [['baz']]])
    y = [2, 2, 2]
    y_expected = [2, 2, 2]
    y_proba_expected = [[1], [1], [1]]
    cls = DummyClassifier(strategy='stratified')
    cls.fit(X, y)
    y_pred = cls.predict(X)
    y_pred_proba = cls.predict_proba(X)
    assert_array_equal(y_pred, y_expected)
    assert_array_equal(y_pred_proba, y_proba_expected)