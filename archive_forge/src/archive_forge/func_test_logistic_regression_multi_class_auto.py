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
@pytest.mark.parametrize('est', [LogisticRegression(random_state=0, max_iter=500), LogisticRegressionCV(random_state=0, cv=3, Cs=3, tol=0.001, max_iter=500)], ids=lambda x: x.__class__.__name__)
@pytest.mark.parametrize('solver', SOLVERS)
def test_logistic_regression_multi_class_auto(est, solver):

    def fit(X, y, **kw):
        return clone(est).set_params(**kw).fit(X, y)
    scaled_data = scale(iris.data)
    X = scaled_data[::10]
    X2 = scaled_data[1::10]
    y_multi = iris.target[::10]
    y_bin = y_multi == 0
    est_auto_bin = fit(X, y_bin, multi_class='auto', solver=solver)
    est_ovr_bin = fit(X, y_bin, multi_class='ovr', solver=solver)
    assert_allclose(est_auto_bin.coef_, est_ovr_bin.coef_)
    assert_allclose(est_auto_bin.predict_proba(X2), est_ovr_bin.predict_proba(X2))
    est_auto_multi = fit(X, y_multi, multi_class='auto', solver=solver)
    if solver in ('liblinear', 'newton-cholesky'):
        est_ovr_multi = fit(X, y_multi, multi_class='ovr', solver=solver)
        assert_allclose(est_auto_multi.coef_, est_ovr_multi.coef_)
        assert_allclose(est_auto_multi.predict_proba(X2), est_ovr_multi.predict_proba(X2))
    else:
        est_multi_multi = fit(X, y_multi, multi_class='multinomial', solver=solver)
        assert_allclose(est_auto_multi.coef_, est_multi_multi.coef_)
        assert_allclose(est_auto_multi.predict_proba(X2), est_multi_multi.predict_proba(X2))
        assert not np.allclose(est_auto_bin.coef_, fit(X, y_bin, multi_class='multinomial', solver=solver).coef_)
        assert not np.allclose(est_auto_bin.coef_, fit(X, y_multi, multi_class='multinomial', solver=solver).coef_)