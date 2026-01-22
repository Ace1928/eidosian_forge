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
@pytest.mark.parametrize('X_train, y_train, X_test', [[X, None, T], [X2, None, T2], [X_blobs[:80], None, X_blobs[80:]], [iris.data, None, iris.data]])
@pytest.mark.parametrize('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
@pytest.mark.parametrize('sparse_container', CSR_CONTAINERS + LIL_CONTAINERS)
@skip_if_32bit
def test_sparse_oneclasssvm(X_train, y_train, X_test, kernel, sparse_container):
    X_train = sparse_container(X_train)
    clf = svm.OneClassSVM(gamma=1, kernel=kernel)
    check_svm_model_equal(clf, X_train, y_train, X_test)