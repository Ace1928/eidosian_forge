import itertools
import math
import pickle
from collections import defaultdict
from functools import partial
from itertools import combinations, product
from typing import Any, Dict
from unittest.mock import patch
import joblib
import numpy as np
import pytest
from scipy.special import comb
import sklearn
from sklearn import clone, datasets
from sklearn.datasets import make_classification, make_hastie_10_2
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
from sklearn.ensemble._forest import (
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree._classes import SPARSE_SPLITTERS
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.parallel import Parallel
from sklearn.utils.validation import check_random_state
@pytest.mark.parametrize('ForestClassifier', FOREST_CLASSIFIERS.values())
def test_classifier_error_oob_score_multiclass_multioutput(ForestClassifier):
    """Check that we raise an error with when requesting OOB score with
    multiclass-multioutput classification target.
    """
    rng = np.random.RandomState(42)
    X = iris.data
    y = rng.randint(low=0, high=5, size=(iris.data.shape[0], 2))
    y_type = type_of_target(y)
    assert y_type == 'multiclass-multioutput'
    estimator = ForestClassifier(oob_score=True, bootstrap=True)
    err_msg = 'The type of target cannot be used to compute OOB estimates'
    with pytest.raises(ValueError, match=err_msg):
        estimator.fit(X, y)