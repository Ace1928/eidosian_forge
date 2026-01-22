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
@pytest.mark.parametrize('ForestRegressor', FOREST_REGRESSORS.values())
def test_forest_multioutput_integral_regression_target(ForestRegressor):
    """Check that multioutput regression with integral values is not interpreted
    as a multiclass-multioutput target and OOB score can be computed.
    """
    rng = np.random.RandomState(42)
    X = iris.data
    y = rng.randint(low=0, high=10, size=(iris.data.shape[0], 2))
    estimator = ForestRegressor(n_estimators=30, oob_score=True, bootstrap=True, random_state=0)
    estimator.fit(X, y)
    n_samples_bootstrap = _get_n_samples_bootstrap(len(X), estimator.max_samples)
    n_samples_test = X.shape[0] // 4
    oob_pred = np.zeros([n_samples_test, 2])
    for sample_idx, sample in enumerate(X[:n_samples_test]):
        n_samples_oob = 0
        oob_pred_sample = np.zeros(2)
        for tree in estimator.estimators_:
            oob_unsampled_indices = _generate_unsampled_indices(tree.random_state, len(X), n_samples_bootstrap)
            if sample_idx in oob_unsampled_indices:
                n_samples_oob += 1
                oob_pred_sample += tree.predict(sample.reshape(1, -1)).squeeze()
        oob_pred[sample_idx] = oob_pred_sample / n_samples_oob
    assert_allclose(oob_pred, estimator.oob_prediction_[:n_samples_test])