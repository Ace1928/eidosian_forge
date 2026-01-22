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
def test_logistic_regression_multinomial():
    n_samples, n_features, n_classes = (50, 20, 3)
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=10, n_classes=n_classes, random_state=0)
    X = StandardScaler(with_mean=False).fit_transform(X)
    solver = 'lbfgs'
    ref_i = LogisticRegression(solver=solver, multi_class='multinomial', tol=1e-06)
    ref_w = LogisticRegression(solver=solver, multi_class='multinomial', fit_intercept=False, tol=1e-06)
    ref_i.fit(X, y)
    ref_w.fit(X, y)
    assert ref_i.coef_.shape == (n_classes, n_features)
    assert ref_w.coef_.shape == (n_classes, n_features)
    for solver in ['sag', 'saga', 'newton-cg']:
        clf_i = LogisticRegression(solver=solver, multi_class='multinomial', random_state=42, max_iter=2000, tol=1e-07)
        clf_w = LogisticRegression(solver=solver, multi_class='multinomial', random_state=42, max_iter=2000, tol=1e-07, fit_intercept=False)
        clf_i.fit(X, y)
        clf_w.fit(X, y)
        assert clf_i.coef_.shape == (n_classes, n_features)
        assert clf_w.coef_.shape == (n_classes, n_features)
        assert_allclose(ref_i.coef_, clf_i.coef_, rtol=0.001)
        assert_allclose(ref_w.coef_, clf_w.coef_, rtol=0.01)
        assert_allclose(ref_i.intercept_, clf_i.intercept_, rtol=0.001)
    for solver in ['lbfgs', 'newton-cg', 'sag', 'saga']:
        clf_path = LogisticRegressionCV(solver=solver, max_iter=2000, tol=1e-06, multi_class='multinomial', Cs=[1.0])
        clf_path.fit(X, y)
        assert_allclose(clf_path.coef_, ref_i.coef_, rtol=0.01)
        assert_allclose(clf_path.intercept_, ref_i.intercept_, rtol=0.01)