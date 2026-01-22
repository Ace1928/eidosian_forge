import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def test_classification_sample_weight():
    X = [[0], [0], [1]]
    y = [0, 1, 0]
    sample_weight = [0.1, 1.0, 0.1]
    clf = DummyClassifier(strategy='stratified').fit(X, y, sample_weight)
    assert_array_almost_equal(clf.class_prior_, [0.2 / 1.2, 1.0 / 1.2])