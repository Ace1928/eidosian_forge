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
def test_precomputed():
    clf = svm.SVC(kernel='precomputed')
    K = np.dot(X, np.array(X).T)
    clf.fit(K, Y)
    KT = np.dot(T, np.array(X).T)
    pred = clf.predict(KT)
    with pytest.raises(ValueError):
        clf.predict(KT.T)
    assert_array_equal(clf.dual_coef_, [[-0.25, 0.25]])
    assert_array_equal(clf.support_, [1, 3])
    assert_array_equal(clf.intercept_, [0])
    assert_array_almost_equal(clf.support_, [1, 3])
    assert_array_equal(pred, true_result)
    KT = np.zeros_like(KT)
    for i in range(len(T)):
        for j in clf.support_:
            KT[i, j] = np.dot(T[i], X[j])
    pred = clf.predict(KT)
    assert_array_equal(pred, true_result)

    def kfunc(x, y):
        return np.dot(x, y.T)
    clf = svm.SVC(kernel=kfunc)
    clf.fit(np.array(X), Y)
    pred = clf.predict(T)
    assert_array_equal(clf.dual_coef_, [[-0.25, 0.25]])
    assert_array_equal(clf.intercept_, [0])
    assert_array_almost_equal(clf.support_, [1, 3])
    assert_array_equal(pred, true_result)
    clf = svm.SVC(kernel='precomputed')
    clf2 = svm.SVC(kernel='linear')
    K = np.dot(iris.data, iris.data.T)
    clf.fit(K, iris.target)
    clf2.fit(iris.data, iris.target)
    pred = clf.predict(K)
    assert_array_almost_equal(clf.support_, clf2.support_)
    assert_array_almost_equal(clf.dual_coef_, clf2.dual_coef_)
    assert_array_almost_equal(clf.intercept_, clf2.intercept_)
    assert_almost_equal(np.mean(pred == iris.target), 0.99, decimal=2)
    K = np.zeros_like(K)
    for i in range(len(iris.data)):
        for j in clf.support_:
            K[i, j] = np.dot(iris.data[i], iris.data[j])
    pred = clf.predict(K)
    assert_almost_equal(np.mean(pred == iris.target), 0.99, decimal=2)
    clf = svm.SVC(kernel=kfunc)
    clf.fit(iris.data, iris.target)
    assert_almost_equal(np.mean(pred == iris.target), 0.99, decimal=2)