import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
@pytest.mark.parametrize('y,y_test', [([2, 1, 1, 1], [2, 2, 1, 1]), (np.array([[2, 2], [1, 1], [1, 1], [1, 1]]), np.array([[2, 2], [2, 2], [1, 1], [1, 1]]))])
def test_classifier_score_with_None(y, y_test):
    clf = DummyClassifier(strategy='most_frequent')
    clf.fit(None, y)
    assert clf.score(None, y_test) == 0.5