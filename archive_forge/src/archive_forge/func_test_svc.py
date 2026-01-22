import numpy as np
import pytest
from scipy import sparse
from sklearn import base, datasets, linear_model, svm
from sklearn.datasets import load_digits, make_blobs, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm.tests import test_svm
from sklearn.utils._testing import (
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.fixes import (
@skip_if_32bit
@pytest.mark.parametrize('X_train, y_train, X_test', [[X, Y, T], [X2, Y2, T2], [X_blobs[:80], y_blobs[:80], X_blobs[80:]], [iris.data, iris.target, iris.data]])
@pytest.mark.parametrize('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
@pytest.mark.parametrize('sparse_container', CSR_CONTAINERS + LIL_CONTAINERS)
def test_svc(X_train, y_train, X_test, kernel, sparse_container):
    """Check that sparse SVC gives the same result as SVC."""
    X_train = sparse_container(X_train)
    clf = svm.SVC(gamma=1, kernel=kernel, probability=True, random_state=0, decision_function_shape='ovo')
    check_svm_model_equal(clf, X_train, y_train, X_test)