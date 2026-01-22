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
@pytest.mark.parametrize('y, groups, expected', [(np.array([0] * 6 + [1] * 6), np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]), np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])), (np.array([0] * 9 + [1] * 3), np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6]), np.asarray([[0.75, 0.25], [0.75, 0.25], [0.75, 0.25]]))])
def test_stratified_group_kfold_homogeneous_groups(y, groups, expected):
    sgkf = StratifiedGroupKFold(n_splits=3)
    X = np.ones_like(y).reshape(-1, 1)
    for (train, test), expect_dist in zip(sgkf.split(X, y, groups), expected):
        assert np.intersect1d(groups[train], groups[test]).size == 0
        split_dist = np.bincount(y[test]) / len(test)
        assert_allclose(split_dist, expect_dist, atol=0.001)