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
@pytest.mark.parametrize('Forest', [RandomForestClassifier, RandomForestRegressor])
def test_missing_value_is_predictive(Forest):
    """Check that the forest learns when missing values are only present for
    a predictive feature."""
    rng = np.random.RandomState(0)
    n_samples = 300
    X_non_predictive = rng.standard_normal(size=(n_samples, 10))
    y = rng.randint(0, high=2, size=n_samples)
    X_random_mask = rng.choice([False, True], size=n_samples, p=[0.95, 0.05])
    y_mask = y.astype(bool)
    y_mask[X_random_mask] = ~y_mask[X_random_mask]
    predictive_feature = rng.standard_normal(size=n_samples)
    predictive_feature[y_mask] = np.nan
    assert np.isnan(predictive_feature).any()
    X_predictive = X_non_predictive.copy()
    X_predictive[:, 5] = predictive_feature
    X_predictive_train, X_predictive_test, X_non_predictive_train, X_non_predictive_test, y_train, y_test = train_test_split(X_predictive, X_non_predictive, y, random_state=0)
    forest_predictive = Forest(random_state=0).fit(X_predictive_train, y_train)
    forest_non_predictive = Forest(random_state=0).fit(X_non_predictive_train, y_train)
    predictive_test_score = forest_predictive.score(X_predictive_test, y_test)
    assert predictive_test_score >= 0.75
    assert predictive_test_score >= forest_non_predictive.score(X_non_predictive_test, y_test)