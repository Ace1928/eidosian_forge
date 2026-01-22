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
@pytest.mark.parametrize('criterion', ('poisson', 'squared_error'))
def test_balance_property_random_forest(criterion):
    """ "Test that sum(y_pred)==sum(y_true) on the training set."""
    rng = np.random.RandomState(42)
    n_train, n_test, n_features = (500, 500, 10)
    X = datasets.make_low_rank_matrix(n_samples=n_train + n_test, n_features=n_features, random_state=rng)
    coef = rng.uniform(low=-2, high=2, size=n_features) / np.max(X, axis=0)
    y = rng.poisson(lam=np.exp(X @ coef))
    reg = RandomForestRegressor(criterion=criterion, n_estimators=10, bootstrap=False, random_state=rng)
    reg.fit(X, y)
    assert np.sum(reg.predict(X)) == pytest.approx(np.sum(y))