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
def test_repeated_kfold_determinstic_split():
    X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    random_state = 258173307
    rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
    for _ in range(3):
        splits = rkf.split(X)
        train, test = next(splits)
        assert_array_equal(train, [2, 4])
        assert_array_equal(test, [0, 1, 3])
        train, test = next(splits)
        assert_array_equal(train, [0, 1, 3])
        assert_array_equal(test, [2, 4])
        train, test = next(splits)
        assert_array_equal(train, [0, 1])
        assert_array_equal(test, [2, 3, 4])
        train, test = next(splits)
        assert_array_equal(train, [2, 3, 4])
        assert_array_equal(test, [0, 1])
        with pytest.raises(StopIteration):
            next(splits)