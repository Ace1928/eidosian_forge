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
@pytest.mark.parametrize('DiscreteNaiveBayes', DISCRETE_NAIVE_BAYES_CLASSES)
def test_discretenb_sample_weight_multiclass(DiscreteNaiveBayes):
    X = [[0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 0, 0]]
    y = [0, 0, 1, 2]
    sample_weight = np.array([1, 1, 2, 2], dtype=np.float64)
    sample_weight /= sample_weight.sum()
    clf = DiscreteNaiveBayes().fit(X, y, sample_weight=sample_weight)
    assert_array_equal(clf.predict(X), [0, 1, 1, 2])
    clf = DiscreteNaiveBayes()
    clf.partial_fit(X[:2], y[:2], classes=[0, 1, 2], sample_weight=sample_weight[:2])
    clf.partial_fit(X[2:3], y[2:3], sample_weight=sample_weight[2:3])
    clf.partial_fit(X[3:], y[3:], sample_weight=sample_weight[3:])
    assert_array_equal(clf.predict(X), [0, 1, 1, 2])