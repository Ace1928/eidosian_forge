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
def test_ovr_decision_function():
    X_train = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    y_train = [0, 1, 2, 3]
    base_points = np.array([[5, 5], [10, 10]])
    X_test = np.vstack((base_points * [1, 1], base_points * [-1, 1], base_points * [-1, -1], base_points * [1, -1]))
    y_test = [0] * 2 + [1] * 2 + [2] * 2 + [3] * 2
    clf = svm.SVC(kernel='linear', decision_function_shape='ovr')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    assert_array_equal(y_pred, y_test)
    deci_val = clf.decision_function(X_test)
    assert_array_equal(np.argmax(deci_val, axis=1), y_pred)
    pred_class_deci_val = deci_val[range(8), y_pred].reshape((4, 2))
    assert np.min(pred_class_deci_val) > 0.0
    assert np.all(pred_class_deci_val[:, 0] < pred_class_deci_val[:, 1])