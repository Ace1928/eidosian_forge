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
@pytest.mark.parametrize('penalty', ('l2', 'elasticnet'))
@pytest.mark.parametrize('multi_class', ('ovr', 'multinomial', 'auto'))
def test_LogisticRegressionCV_no_refit(penalty, multi_class):
    n_classes = 3
    n_features = 20
    X, y = make_classification(n_samples=200, n_classes=n_classes, n_informative=n_classes, n_features=n_features, random_state=0)
    Cs = np.logspace(-4, 4, 3)
    if penalty == 'elasticnet':
        l1_ratios = np.linspace(0, 1, 2)
    else:
        l1_ratios = None
    lrcv = LogisticRegressionCV(penalty=penalty, Cs=Cs, solver='saga', l1_ratios=l1_ratios, random_state=0, multi_class=multi_class, tol=0.01, refit=False)
    lrcv.fit(X, y)
    assert lrcv.C_.shape == (n_classes,)
    assert lrcv.l1_ratio_.shape == (n_classes,)
    assert lrcv.coef_.shape == (n_classes, n_features)