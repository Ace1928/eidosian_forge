import itertools
import os
import warnings
from functools import partial
import numpy as np
import pytest
from numpy.testing import (
from scipy import sparse
from sklearn import config_context
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.metrics import get_scorer, log_loss
from sklearn.model_selection import (
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale
from sklearn.svm import l1_min_c
from sklearn.utils import _IS_32BIT, compute_class_weight, shuffle
from sklearn.utils._testing import ignore_warnings, skip_if_no_parallel
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('weight', [{0: 0.1, 1: 0.2}, {0: 0.1, 1: 0.2, 2: 0.5}])
@pytest.mark.parametrize('class_weight', ['weight', 'balanced'])
def test_logistic_regressioncv_class_weights(weight, class_weight, global_random_seed):
    """Test class_weight for LogisticRegressionCV."""
    n_classes = len(weight)
    if class_weight == 'weight':
        class_weight = weight
    X, y = make_classification(n_samples=30, n_features=3, n_repeated=0, n_informative=3, n_redundant=0, n_classes=n_classes, random_state=global_random_seed)
    params = dict(Cs=1, fit_intercept=False, multi_class='ovr', class_weight=class_weight, tol=1e-08)
    clf_lbfgs = LogisticRegressionCV(solver='lbfgs', **params)
    with ignore_warnings(category=ConvergenceWarning):
        clf_lbfgs.fit(X, y)
    for solver in set(SOLVERS) - set(['lbfgs']):
        clf = LogisticRegressionCV(solver=solver, **params)
        if solver in ('sag', 'saga'):
            clf.set_params(tol=1e-18, max_iter=10000, random_state=global_random_seed + 1)
        clf.fit(X, y)
        assert_allclose(clf.coef_, clf_lbfgs.coef_, rtol=0.001, err_msg=f'{solver} vs lbfgs')