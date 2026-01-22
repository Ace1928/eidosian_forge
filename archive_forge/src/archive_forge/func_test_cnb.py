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
def test_cnb():
    X = np.array([[1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0], [0, 1, 1, 0, 0, 1]])
    Y = np.array([0, 0, 0, 1])
    theta = np.array([[(0 + 1) / (3 + 6), (1 + 1) / (3 + 6), (1 + 1) / (3 + 6), (0 + 1) / (3 + 6), (0 + 1) / (3 + 6), (1 + 1) / (3 + 6)], [(1 + 1) / (6 + 6), (3 + 1) / (6 + 6), (0 + 1) / (6 + 6), (1 + 1) / (6 + 6), (1 + 1) / (6 + 6), (0 + 1) / (6 + 6)]])
    weights = np.zeros(theta.shape)
    normed_weights = np.zeros(theta.shape)
    for i in range(2):
        weights[i] = -np.log(theta[i])
        normed_weights[i] = weights[i] / weights[i].sum()
    clf = ComplementNB(alpha=1.0)
    msg = re.escape('Negative values in data passed to ComplementNB (input X)')
    with pytest.raises(ValueError, match=msg):
        clf.fit(-X, Y)
    clf.fit(X, Y)
    feature_count = np.array([[1, 3, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1]])
    assert_array_equal(clf.feature_count_, feature_count)
    class_count = np.array([3, 1])
    assert_array_equal(clf.class_count_, class_count)
    feature_all = np.array([1, 4, 1, 1, 1, 1])
    assert_array_equal(clf.feature_all_, feature_all)
    assert_array_almost_equal(clf.feature_log_prob_, weights)
    clf = ComplementNB(alpha=1.0, norm=True)
    clf.fit(X, Y)
    assert_array_almost_equal(clf.feature_log_prob_, normed_weights)