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
@pytest.mark.parametrize('lil_container', LIL_CONTAINERS)
def test_sparse_svc_clone_with_callable_kernel(lil_container):
    a = svm.SVC(C=1, kernel=lambda x, y: x @ y.T, probability=True, random_state=0)
    b = base.clone(a)
    X_sp = lil_container(X)
    b.fit(X_sp, Y)
    pred = b.predict(X_sp)
    b.predict_proba(X_sp)
    dense_svm = svm.SVC(C=1, kernel=lambda x, y: np.dot(x, y.T), probability=True, random_state=0)
    pred_dense = dense_svm.fit(X, Y).predict(X)
    assert_array_equal(pred_dense, pred)