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
@pytest.mark.parametrize('solver', sorted(set(SOLVERS) - set(['liblinear'])))
def test_penalty_none(solver):
    X, y = make_classification(n_samples=1000, n_redundant=0, random_state=0)
    msg = 'Setting penalty=None will ignore the C'
    lr = LogisticRegression(penalty=None, solver=solver, C=4)
    with pytest.warns(UserWarning, match=msg):
        lr.fit(X, y)
    lr_none = LogisticRegression(penalty=None, solver=solver, random_state=0)
    lr_l2_C_inf = LogisticRegression(penalty='l2', C=np.inf, solver=solver, random_state=0)
    pred_none = lr_none.fit(X, y).predict(X)
    pred_l2_C_inf = lr_l2_C_inf.fit(X, y).predict(X)
    assert_array_equal(pred_none, pred_l2_C_inf)