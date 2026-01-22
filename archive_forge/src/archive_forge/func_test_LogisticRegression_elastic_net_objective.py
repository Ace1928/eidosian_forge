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
@pytest.mark.parametrize('C', np.logspace(-3, 2, 4))
@pytest.mark.parametrize('l1_ratio', [0.1, 0.5, 0.9])
def test_LogisticRegression_elastic_net_objective(C, l1_ratio):
    X, y = make_classification(n_samples=1000, n_classes=2, n_features=20, n_informative=10, n_redundant=0, n_repeated=0, random_state=0)
    X = scale(X)
    lr_enet = LogisticRegression(penalty='elasticnet', solver='saga', random_state=0, C=C, l1_ratio=l1_ratio, fit_intercept=False)
    lr_l2 = LogisticRegression(penalty='l2', solver='saga', random_state=0, C=C, fit_intercept=False)
    lr_enet.fit(X, y)
    lr_l2.fit(X, y)

    def enet_objective(lr):
        coef = lr.coef_.ravel()
        obj = C * log_loss(y, lr.predict_proba(X))
        obj += l1_ratio * np.sum(np.abs(coef))
        obj += (1.0 - l1_ratio) * 0.5 * np.dot(coef, coef)
        return obj
    assert enet_objective(lr_enet) < enet_objective(lr_l2)