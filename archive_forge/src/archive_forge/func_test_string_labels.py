import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def test_string_labels():
    X = [[0]] * 5
    y = ['paris', 'paris', 'tokyo', 'amsterdam', 'berlin']
    clf = DummyClassifier(strategy='most_frequent')
    clf.fit(X, y)
    assert_array_equal(clf.predict(X), ['paris'] * 5)