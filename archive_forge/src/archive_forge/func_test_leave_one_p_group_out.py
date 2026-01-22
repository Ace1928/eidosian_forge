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
def test_leave_one_p_group_out():
    logo = LeaveOneGroupOut()
    lpgo_1 = LeavePGroupsOut(n_groups=1)
    lpgo_2 = LeavePGroupsOut(n_groups=2)
    assert repr(logo) == 'LeaveOneGroupOut()'
    assert repr(lpgo_1) == 'LeavePGroupsOut(n_groups=1)'
    assert repr(lpgo_2) == 'LeavePGroupsOut(n_groups=2)'
    assert repr(LeavePGroupsOut(n_groups=3)) == 'LeavePGroupsOut(n_groups=3)'
    for j, (cv, p_groups_out) in enumerate(((logo, 1), (lpgo_1, 1), (lpgo_2, 2))):
        for i, groups_i in enumerate(test_groups):
            n_groups = len(np.unique(groups_i))
            n_splits = n_groups if p_groups_out == 1 else n_groups * (n_groups - 1) / 2
            X = y = np.ones(len(groups_i))
            assert cv.get_n_splits(X, y, groups=groups_i) == n_splits
            groups_arr = np.asarray(groups_i)
            for train, test in cv.split(X, y, groups=groups_i):
                assert_array_equal(np.intersect1d(groups_arr[train], groups_arr[test]).tolist(), [])
                assert len(train) + len(test) == len(groups_i)
                assert np.unique(groups_arr[test]).shape[0], p_groups_out
    assert logo.get_n_splits(None, None, ['a', 'b', 'c', 'b', 'c']) == 3
    assert logo.get_n_splits(groups=[1.0, 1.1, 1.0, 1.2]) == 3
    assert lpgo_2.get_n_splits(None, None, np.arange(4)) == 6
    assert lpgo_1.get_n_splits(groups=np.arange(4)) == 4
    with pytest.raises(ValueError):
        logo.get_n_splits(None, None, [0.0, np.nan, 0.0])
    with pytest.raises(ValueError):
        lpgo_2.get_n_splits(None, None, [0.0, np.inf, 0.0])
    msg = "The 'groups' parameter should not be None."
    with pytest.raises(ValueError, match=msg):
        logo.get_n_splits(None, None, None)
    with pytest.raises(ValueError, match=msg):
        lpgo_1.get_n_splits(None, None, None)