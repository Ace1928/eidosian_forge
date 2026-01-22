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
def test_elastic_net_versus_sgd(C, l1_ratio):
    n_samples = 500
    X, y = make_classification(n_samples=n_samples, n_classes=2, n_features=5, n_informative=5, n_redundant=0, n_repeated=0, random_state=1)
    X = scale(X)
    sgd = SGDClassifier(penalty='elasticnet', random_state=1, fit_intercept=False, tol=None, max_iter=2000, l1_ratio=l1_ratio, alpha=1.0 / C / n_samples, loss='log_loss')
    log = LogisticRegression(penalty='elasticnet', random_state=1, fit_intercept=False, tol=1e-05, max_iter=1000, l1_ratio=l1_ratio, C=C, solver='saga')
    sgd.fit(X, y)
    log.fit(X, y)
    assert_array_almost_equal(sgd.coef_, log.coef_, decimal=1)