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
@pytest.mark.parametrize('multi_class', ('ovr', 'multinomial'))
def test_LogisticRegressionCV_GridSearchCV_elastic_net(multi_class):
    if multi_class == 'ovr':
        X, y = make_classification(random_state=0)
    else:
        X, y = make_classification(n_samples=100, n_classes=3, n_informative=3, random_state=0)
    cv = StratifiedKFold(5)
    l1_ratios = np.linspace(0, 1, 3)
    Cs = np.logspace(-4, 4, 3)
    lrcv = LogisticRegressionCV(penalty='elasticnet', Cs=Cs, solver='saga', cv=cv, l1_ratios=l1_ratios, random_state=0, multi_class=multi_class, tol=0.01)
    lrcv.fit(X, y)
    param_grid = {'C': Cs, 'l1_ratio': l1_ratios}
    lr = LogisticRegression(penalty='elasticnet', solver='saga', random_state=0, multi_class=multi_class, tol=0.01)
    gs = GridSearchCV(lr, param_grid, cv=cv)
    gs.fit(X, y)
    assert gs.best_params_['l1_ratio'] == lrcv.l1_ratio_[0]
    assert gs.best_params_['C'] == lrcv.C_[0]