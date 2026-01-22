import re
from math import sqrt
import numpy as np
import pytest
from sklearn import metrics, neighbors
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_hasattr_prediction():
    X = [[1, 1], [1, 2], [2, 1]]
    clf = neighbors.LocalOutlierFactor(novelty=True)
    clf.fit(X)
    assert hasattr(clf, 'predict')
    assert hasattr(clf, 'decision_function')
    assert hasattr(clf, 'score_samples')
    assert not hasattr(clf, 'fit_predict')
    clf = neighbors.LocalOutlierFactor(novelty=False)
    clf.fit(X)
    assert hasattr(clf, 'fit_predict')
    assert not hasattr(clf, 'predict')
    assert not hasattr(clf, 'decision_function')
    assert not hasattr(clf, 'score_samples')