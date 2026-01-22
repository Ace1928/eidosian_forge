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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_sparse_decision_function(csr_container):
    iris_data_sp = csr_container(iris.data)
    svc = svm.SVC(kernel='linear', C=0.1, decision_function_shape='ovo')
    clf = svc.fit(iris_data_sp, iris.target)
    dec = safe_sparse_dot(iris_data_sp, clf.coef_.T) + clf.intercept_
    assert_allclose(dec, clf.decision_function(iris_data_sp))
    clf.fit(X, Y)
    dec = np.dot(X, clf.coef_.T) + clf.intercept_
    prediction = clf.predict(X)
    assert_allclose(dec.ravel(), clf.decision_function(X))
    assert_allclose(prediction, clf.classes_[(clf.decision_function(X) > 0).astype(int).ravel()])
    expected = np.array([-1.0, -0.66, -1.0, 0.66, 1.0, 1.0])
    assert_array_almost_equal(clf.decision_function(X), expected, decimal=2)