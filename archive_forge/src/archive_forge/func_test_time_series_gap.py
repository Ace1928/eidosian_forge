import re
import warnings
from itertools import combinations, combinations_with_replacement, permutations
import numpy as np
import pytest
from scipy import stats
from scipy.sparse import issparse
from scipy.special import comb
from sklearn import config_context
from sklearn.datasets import load_digits, make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
from sklearn.model_selection._split import (
from sklearn.svm import SVC
from sklearn.tests.metadata_routing_common import assert_request_is_empty
from sklearn.utils._array_api import (
from sklearn.utils._array_api import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
def test_time_series_gap():
    X = np.zeros((10, 1))
    splits = TimeSeriesSplit(n_splits=2, gap=2).split(X)
    train, test = next(splits)
    assert_array_equal(train, [0, 1])
    assert_array_equal(test, [4, 5, 6])
    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3, 4])
    assert_array_equal(test, [7, 8, 9])
    splits = TimeSeriesSplit(n_splits=3, gap=2, max_train_size=2).split(X)
    train, test = next(splits)
    assert_array_equal(train, [0, 1])
    assert_array_equal(test, [4, 5])
    train, test = next(splits)
    assert_array_equal(train, [2, 3])
    assert_array_equal(test, [6, 7])
    train, test = next(splits)
    assert_array_equal(train, [4, 5])
    assert_array_equal(test, [8, 9])
    splits = TimeSeriesSplit(n_splits=2, gap=2, max_train_size=4, test_size=2).split(X)
    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3])
    assert_array_equal(test, [6, 7])
    train, test = next(splits)
    assert_array_equal(train, [2, 3, 4, 5])
    assert_array_equal(test, [8, 9])
    splits = TimeSeriesSplit(n_splits=2, gap=2, test_size=3).split(X)
    train, test = next(splits)
    assert_array_equal(train, [0, 1])
    assert_array_equal(test, [4, 5, 6])
    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3, 4])
    assert_array_equal(test, [7, 8, 9])
    with pytest.raises(ValueError, match='Too many splits.*and gap'):
        splits = TimeSeriesSplit(n_splits=4, gap=2).split(X)
        next(splits)