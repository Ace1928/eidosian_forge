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
@pytest.mark.parametrize('multi_class', ['ovr', 'multinomial', 'auto'])
@pytest.mark.parametrize('class_weight', [{0: 1.0, 1: 10.0, 2: 1.0}, 'balanced'])
def test_sample_weight_not_modified(multi_class, class_weight):
    X, y = load_iris(return_X_y=True)
    n_features = len(X)
    W = np.ones(n_features)
    W[:n_features // 2] = 2
    expected = W.copy()
    clf = LogisticRegression(random_state=0, class_weight=class_weight, max_iter=200, multi_class=multi_class)
    clf.fit(X, y, sample_weight=W)
    assert_allclose(expected, W)