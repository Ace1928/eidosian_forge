import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.neighbors import NearestCentroid
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_classification_toy(csr_container):
    X_csr = csr_container(X)
    T_csr = csr_container(T)
    clf = NearestCentroid()
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)
    clf = NearestCentroid()
    clf.fit(X_csr, y)
    assert_array_equal(clf.predict(T_csr), true_result)
    clf = NearestCentroid()
    clf.fit(X_csr, y)
    assert_array_equal(clf.predict(T), true_result)
    clf = NearestCentroid()
    clf.fit(X, y)
    assert_array_equal(clf.predict(T_csr), true_result)
    clf = NearestCentroid()
    clf.fit(X_csr.tocoo(), y)
    assert_array_equal(clf.predict(T_csr.tolil()), true_result)