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
def test_2d_y():
    n_samples = 30
    rng = np.random.RandomState(1)
    X = rng.randint(0, 3, size=(n_samples, 2))
    y = rng.randint(0, 3, size=(n_samples,))
    y_2d = y.reshape(-1, 1)
    y_multilabel = rng.randint(0, 2, size=(n_samples, 3))
    groups = rng.randint(0, 3, size=(n_samples,))
    splitters = [LeaveOneOut(), LeavePOut(p=2), KFold(), StratifiedKFold(), RepeatedKFold(), RepeatedStratifiedKFold(), StratifiedGroupKFold(), ShuffleSplit(), StratifiedShuffleSplit(test_size=0.5), GroupShuffleSplit(), LeaveOneGroupOut(), LeavePGroupsOut(n_groups=2), GroupKFold(n_splits=3), TimeSeriesSplit(), PredefinedSplit(test_fold=groups)]
    for splitter in splitters:
        list(splitter.split(X, y, groups))
        list(splitter.split(X, y_2d, groups))
        try:
            list(splitter.split(X, y_multilabel, groups))
        except ValueError as e:
            allowed_target_types = ('binary', 'multiclass')
            msg = "Supported target types are: {}. Got 'multilabel".format(allowed_target_types)
            assert msg in str(e)