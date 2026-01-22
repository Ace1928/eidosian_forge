import numpy as np
import pytest
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('average', [False, True])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('csr_container', [None, *CSR_CONTAINERS])
def test_classifier_accuracy(csr_container, fit_intercept, average):
    data = csr_container(X) if csr_container is not None else X
    clf = PassiveAggressiveClassifier(C=1.0, max_iter=30, fit_intercept=fit_intercept, random_state=1, average=average, tol=None)
    clf.fit(data, y)
    score = clf.score(data, y)
    assert score > 0.79
    if average:
        assert hasattr(clf, '_average_coef')
        assert hasattr(clf, '_average_intercept')
        assert hasattr(clf, '_standard_intercept')
        assert hasattr(clf, '_standard_coef')