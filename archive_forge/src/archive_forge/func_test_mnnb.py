import re
import warnings
import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('kind', ('dense', 'sparse'))
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_mnnb(kind, global_random_seed, csr_container):
    X2, y2 = get_random_integer_x_three_classes_y(global_random_seed)
    if kind == 'dense':
        X = X2
    elif kind == 'sparse':
        X = csr_container(X2)
    clf = MultinomialNB()
    msg = 'Negative values in data passed to'
    with pytest.raises(ValueError, match=msg):
        clf.fit(-X, y2)
    y_pred = clf.fit(X, y2).predict(X)
    assert_array_equal(y_pred, y2)
    y_pred_proba = clf.predict_proba(X)
    y_pred_log_proba = clf.predict_log_proba(X)
    assert_array_almost_equal(np.log(y_pred_proba), y_pred_log_proba, 8)
    clf2 = MultinomialNB()
    clf2.partial_fit(X[:2], y2[:2], classes=np.unique(y2))
    clf2.partial_fit(X[2:5], y2[2:5])
    clf2.partial_fit(X[5:], y2[5:])
    y_pred2 = clf2.predict(X)
    assert_array_equal(y_pred2, y2)
    y_pred_proba2 = clf2.predict_proba(X)
    y_pred_log_proba2 = clf2.predict_log_proba(X)
    assert_array_almost_equal(np.log(y_pred_proba2), y_pred_log_proba2, 8)
    assert_array_almost_equal(y_pred_proba2, y_pred_proba)
    assert_array_almost_equal(y_pred_log_proba2, y_pred_log_proba)
    clf3 = MultinomialNB()
    clf3.partial_fit(X, y2, classes=np.unique(y2))
    y_pred3 = clf3.predict(X)
    assert_array_equal(y_pred3, y2)
    y_pred_proba3 = clf3.predict_proba(X)
    y_pred_log_proba3 = clf3.predict_log_proba(X)
    assert_array_almost_equal(np.log(y_pred_proba3), y_pred_log_proba3, 8)
    assert_array_almost_equal(y_pred_proba3, y_pred_proba)
    assert_array_almost_equal(y_pred_log_proba3, y_pred_log_proba)