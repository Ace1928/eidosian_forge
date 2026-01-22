import re
import numpy as np
import pytest
from numpy.testing import (
from sklearn import base, datasets, linear_model, metrics, svm
from sklearn.datasets import make_blobs, make_classification
from sklearn.exceptions import (
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import (  # type: ignore
from sklearn.svm._classes import _validate_dual_parameter
from sklearn.utils import check_random_state, shuffle
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.validation import _num_samples
def test_oneclass_decision_function():
    clf = svm.OneClassSVM()
    rnd = check_random_state(2)
    X = 0.3 * rnd.randn(100, 2)
    X_train = np.r_[X + 2, X - 2]
    X = 0.3 * rnd.randn(20, 2)
    X_test = np.r_[X + 2, X - 2]
    X_outliers = rnd.uniform(low=-4, high=4, size=(20, 2))
    clf = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)
    clf.fit(X_train)
    y_pred_test = clf.predict(X_test)
    assert np.mean(y_pred_test == 1) > 0.9
    y_pred_outliers = clf.predict(X_outliers)
    assert np.mean(y_pred_outliers == -1) > 0.9
    dec_func_test = clf.decision_function(X_test)
    assert_array_equal((dec_func_test > 0).ravel(), y_pred_test == 1)
    dec_func_outliers = clf.decision_function(X_outliers)
    assert_array_equal((dec_func_outliers > 0).ravel(), y_pred_outliers == 1)