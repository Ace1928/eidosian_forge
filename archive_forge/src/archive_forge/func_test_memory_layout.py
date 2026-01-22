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
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS_REGRESSORS)
@pytest.mark.parametrize('dtype', (np.float64, np.float32))
def test_memory_layout(name, dtype):
    est = FOREST_ESTIMATORS[name](random_state=0, bootstrap=False)
    for container, kwargs in ((np.asarray, {}), (np.asarray, {'order': 'C'}), (np.asarray, {'order': 'F'}), (np.ascontiguousarray, {})):
        X = container(iris.data, dtype=dtype, **kwargs)
        y = iris.target
        assert_array_almost_equal(est.fit(X, y).predict(X), y)
    if est.estimator.splitter in SPARSE_SPLITTERS:
        for sparse_container in COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS:
            X = sparse_container(iris.data, dtype=dtype)
            y = iris.target
            assert_array_almost_equal(est.fit(X, y).predict(X), y)
    X = np.asarray(iris.data[::3], dtype=dtype)
    y = iris.target[::3]
    assert_array_almost_equal(est.fit(X, y).predict(X), y)