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
def test_logistic_regression_solvers():
    """Test solvers converge to the same result."""
    X, y = make_classification(n_features=10, n_informative=5, random_state=0)
    params = dict(fit_intercept=False, random_state=42, multi_class='ovr')
    regressors = {solver: LogisticRegression(solver=solver, **params).fit(X, y) for solver in SOLVERS}
    for solver_1, solver_2 in itertools.combinations(regressors, r=2):
        assert_array_almost_equal(regressors[solver_1].coef_, regressors[solver_2].coef_, decimal=3)