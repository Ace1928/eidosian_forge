from math import ceil
import numpy as np
import pytest
from scipy.stats import expon, norm, randint
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
from sklearn.model_selection._search_successive_halving import (
from sklearn.model_selection.tests.test_search import (
from sklearn.svm import SVC, LinearSVC
@pytest.mark.parametrize('subsample_test', (True, False))
def test_subsample_splitter_determinism(subsample_test):
    n_samples = 100
    X, y = make_classification(n_samples)
    cv = _SubsampleMetaSplitter(base_cv=KFold(5), fraction=0.5, subsample_test=subsample_test, random_state=None)
    folds_a = list(cv.split(X, y, groups=None))
    folds_b = list(cv.split(X, y, groups=None))
    for (train_a, test_a), (train_b, test_b) in zip(folds_a, folds_b):
        assert not np.all(train_a == train_b)
        if subsample_test:
            assert not np.all(test_a == test_b)
        else:
            assert np.all(test_a == test_b)
            assert np.all(X[test_a] == X[test_b])