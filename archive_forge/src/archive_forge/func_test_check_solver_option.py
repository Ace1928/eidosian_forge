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
@pytest.mark.parametrize('LR', [LogisticRegression, LogisticRegressionCV])
def test_check_solver_option(LR):
    X, y = (iris.data, iris.target)
    for solver in ['liblinear', 'newton-cholesky']:
        msg = f'Solver {solver} does not support a multinomial backend.'
        lr = LR(solver=solver, multi_class='multinomial')
        with pytest.raises(ValueError, match=msg):
            lr.fit(X, y)
    for solver in ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag']:
        msg = "Solver %s supports only 'l2' or None penalties," % solver
        lr = LR(solver=solver, penalty='l1', multi_class='ovr')
        with pytest.raises(ValueError, match=msg):
            lr.fit(X, y)
    for solver in ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga']:
        msg = 'Solver %s supports only dual=False, got dual=True' % solver
        lr = LR(solver=solver, dual=True, multi_class='ovr')
        with pytest.raises(ValueError, match=msg):
            lr.fit(X, y)
    for solver in ['liblinear']:
        msg = f"Only 'saga' solver supports elasticnet penalty, got solver={solver}."
        lr = LR(solver=solver, penalty='elasticnet')
        with pytest.raises(ValueError, match=msg):
            lr.fit(X, y)
    if LR is LogisticRegression:
        msg = 'penalty=None is not supported for the liblinear solver'
        lr = LR(penalty=None, solver='liblinear')
        with pytest.raises(ValueError, match=msg):
            lr.fit(X, y)