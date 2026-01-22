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
def test_mnb_prior_unobserved_targets():
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])
    clf = MultinomialNB()
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        clf.partial_fit(X, y, classes=[0, 1, 2])
    assert clf.predict([[0, 1]]) == 0
    assert clf.predict([[1, 0]]) == 1
    assert clf.predict([[1, 1]]) == 0
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        clf.partial_fit([[1, 1]], [2])
    assert clf.predict([[0, 1]]) == 0
    assert clf.predict([[1, 0]]) == 1
    assert clf.predict([[1, 1]]) == 2