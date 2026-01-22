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
@pytest.mark.filterwarnings('ignore:The max_iter was reached')
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_multiclass_classifier_class_weight(csr_container):
    """tests multiclass with classweights for each class"""
    alpha = 0.1
    n_samples = 20
    tol = 1e-05
    max_iter = 50
    class_weight = {0: 0.45, 1: 0.55, 2: 0.75}
    fit_intercept = True
    X, y = make_blobs(n_samples=n_samples, centers=3, random_state=0, cluster_std=0.1)
    step_size = get_step_size(X, alpha, fit_intercept, classification=True)
    classes = np.unique(y)
    clf1 = LogisticRegression(solver='sag', C=1.0 / alpha / n_samples, max_iter=max_iter, tol=tol, random_state=77, fit_intercept=fit_intercept, multi_class='ovr', class_weight=class_weight)
    clf2 = clone(clf1)
    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)
    le = LabelEncoder()
    class_weight_ = compute_class_weight(class_weight, classes=np.unique(y), y=y)
    sample_weight = class_weight_[le.fit_transform(y)]
    coef1 = []
    intercept1 = []
    coef2 = []
    intercept2 = []
    for cl in classes:
        y_encoded = np.ones(n_samples)
        y_encoded[y != cl] = -1
        spweights1, spintercept1 = sag_sparse(X, y_encoded, step_size, alpha, n_iter=max_iter, dloss=log_dloss, sample_weight=sample_weight)
        spweights2, spintercept2 = sag_sparse(X, y_encoded, step_size, alpha, n_iter=max_iter, dloss=log_dloss, sample_weight=sample_weight, sparse=True)
        coef1.append(spweights1)
        intercept1.append(spintercept1)
        coef2.append(spweights2)
        intercept2.append(spintercept2)
    coef1 = np.vstack(coef1)
    intercept1 = np.array(intercept1)
    coef2 = np.vstack(coef2)
    intercept2 = np.array(intercept2)
    for i, cl in enumerate(classes):
        assert_array_almost_equal(clf1.coef_[i].ravel(), coef1[i].ravel(), decimal=2)
        assert_almost_equal(clf1.intercept_[i], intercept1[i], decimal=1)
        assert_array_almost_equal(clf2.coef_[i].ravel(), coef2[i].ravel(), decimal=2)
        assert_almost_equal(clf2.intercept_[i], intercept2[i], decimal=1)