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
def test_liblinear_not_stuck():
    X = iris.data.copy()
    y = iris.target.copy()
    X = X[y != 2]
    y = y[y != 2]
    X_prep = StandardScaler().fit_transform(X)
    C = l1_min_c(X, y, loss='log') * 10 ** (10 / 29)
    clf = LogisticRegression(penalty='l1', solver='liblinear', tol=1e-06, max_iter=100, intercept_scaling=10000.0, random_state=0, C=C)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        clf.fit(X_prep, y)