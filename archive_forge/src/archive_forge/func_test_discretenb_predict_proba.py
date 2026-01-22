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
def test_discretenb_predict_proba():
    X_bernoulli = [[1, 100, 0], [0, 1, 0], [0, 100, 1]]
    X_multinomial = [[0, 1], [1, 3], [4, 0]]
    y = [0, 0, 2]
    for DiscreteNaiveBayes, X in zip([BernoulliNB, MultinomialNB], [X_bernoulli, X_multinomial]):
        clf = DiscreteNaiveBayes().fit(X, y)
        assert clf.predict(X[-1:]) == 2
        assert clf.predict_proba([X[0]]).shape == (1, 2)
        assert_array_almost_equal(clf.predict_proba(X[:2]).sum(axis=1), np.array([1.0, 1.0]), 6)
    y = [0, 1, 2]
    for DiscreteNaiveBayes, X in zip([BernoulliNB, MultinomialNB], [X_bernoulli, X_multinomial]):
        clf = DiscreteNaiveBayes().fit(X, y)
        assert clf.predict_proba(X[0:1]).shape == (1, 3)
        assert clf.predict_proba(X[:2]).shape == (2, 3)
        assert_almost_equal(np.sum(clf.predict_proba([X[1]])), 1)
        assert_almost_equal(np.sum(clf.predict_proba([X[-1]])), 1)
        assert_almost_equal(np.sum(np.exp(clf.class_log_prior_)), 1)