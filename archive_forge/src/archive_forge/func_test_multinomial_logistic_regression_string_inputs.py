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
def test_multinomial_logistic_regression_string_inputs():
    n_samples, n_features, n_classes = (50, 5, 3)
    X_ref, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_informative=3, random_state=0)
    y_str = LabelEncoder().fit(['bar', 'baz', 'foo']).inverse_transform(y)
    y = np.array(y) - 1
    lr = LogisticRegression(multi_class='multinomial')
    lr_cv = LogisticRegressionCV(multi_class='multinomial', Cs=3)
    lr_str = LogisticRegression(multi_class='multinomial')
    lr_cv_str = LogisticRegressionCV(multi_class='multinomial', Cs=3)
    lr.fit(X_ref, y)
    lr_cv.fit(X_ref, y)
    lr_str.fit(X_ref, y_str)
    lr_cv_str.fit(X_ref, y_str)
    assert_array_almost_equal(lr.coef_, lr_str.coef_)
    assert sorted(lr_str.classes_) == ['bar', 'baz', 'foo']
    assert_array_almost_equal(lr_cv.coef_, lr_cv_str.coef_)
    assert sorted(lr_str.classes_) == ['bar', 'baz', 'foo']
    assert sorted(lr_cv_str.classes_) == ['bar', 'baz', 'foo']
    assert sorted(np.unique(lr_str.predict(X_ref))) == ['bar', 'baz', 'foo']
    assert sorted(np.unique(lr_cv_str.predict(X_ref))) == ['bar', 'baz', 'foo']
    lr_cv_str = LogisticRegression(class_weight={'bar': 1, 'baz': 2, 'foo': 0}, multi_class='multinomial').fit(X_ref, y_str)
    assert sorted(np.unique(lr_cv_str.predict(X_ref))) == ['bar', 'baz']