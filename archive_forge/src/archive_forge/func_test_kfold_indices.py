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
def test_kfold_indices():
    X1 = np.ones(18)
    kf = KFold(3)
    check_cv_coverage(kf, X1, y=None, groups=None, expected_n_splits=3)
    X2 = np.ones(17)
    kf = KFold(3)
    check_cv_coverage(kf, X2, y=None, groups=None, expected_n_splits=3)
    assert 5 == KFold(5).get_n_splits(X2)