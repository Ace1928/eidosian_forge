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
@pytest.mark.parametrize('random_seed', [42])
@pytest.mark.parametrize('penalty', ['l1', 'l2'])
def test_logistic_regression_cv_refit(random_seed, penalty):
    X, y = make_classification(n_samples=100, n_features=20, random_state=random_seed)
    common_params = dict(solver='saga', penalty=penalty, random_state=random_seed, max_iter=1000, tol=1e-12)
    lr_cv = LogisticRegressionCV(Cs=[1.0], refit=True, **common_params)
    lr_cv.fit(X, y)
    lr = LogisticRegression(C=1.0, **common_params)
    lr.fit(X, y)
    assert_array_almost_equal(lr_cv.coef_, lr.coef_)