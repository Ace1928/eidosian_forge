import math
import re
import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn._loss.loss import HalfMultinomialLoss
from sklearn.base import clone
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.linear_model._base import make_dataset
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.linear_model._sag import get_auto_step_size
from sklearn.linear_model._sag_fast import _multinomial_grad_loss_all_samples
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import check_random_state, compute_class_weight
from sklearn.utils._testing import (
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import CSR_CONTAINERS
def test_classifier_matching():
    n_samples = 20
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=0, cluster_std=0.1)
    y[y == 0] = -1
    alpha = 1.1
    fit_intercept = True
    step_size = get_step_size(X, alpha, fit_intercept)
    for solver in ['sag', 'saga']:
        if solver == 'sag':
            n_iter = 80
        else:
            n_iter = 300
        clf = LogisticRegression(solver=solver, fit_intercept=fit_intercept, tol=1e-11, C=1.0 / alpha / n_samples, max_iter=n_iter, random_state=10, multi_class='ovr')
        clf.fit(X, y)
        weights, intercept = sag_sparse(X, y, step_size, alpha, n_iter=n_iter, dloss=log_dloss, fit_intercept=fit_intercept, saga=solver == 'saga')
        weights2, intercept2 = sag(X, y, step_size, alpha, n_iter=n_iter, dloss=log_dloss, fit_intercept=fit_intercept, saga=solver == 'saga')
        weights = np.atleast_2d(weights)
        intercept = np.atleast_1d(intercept)
        weights2 = np.atleast_2d(weights2)
        intercept2 = np.atleast_1d(intercept2)
        assert_array_almost_equal(weights, clf.coef_, decimal=9)
        assert_array_almost_equal(intercept, clf.intercept_, decimal=9)
        assert_array_almost_equal(weights2, clf.coef_, decimal=9)
        assert_array_almost_equal(intercept2, clf.intercept_, decimal=9)