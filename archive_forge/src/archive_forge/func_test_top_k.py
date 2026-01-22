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
@pytest.mark.parametrize('k, itr, expected', [(1, 0, ['c']), (2, 0, ['a', 'c']), (4, 0, ['d', 'b', 'a', 'c']), (10, 0, ['d', 'b', 'a', 'c']), (1, 1, ['e']), (2, 1, ['f', 'e']), (10, 1, ['f', 'e']), (1, 2, ['i']), (10, 2, ['g', 'h', 'i'])])
def test_top_k(k, itr, expected):
    results = {'iter': [0, 0, 0, 0, 1, 1, 2, 2, 2], 'mean_test_score': [4, 3, 5, 1, 11, 10, 5, 6, 9], 'params': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']}
    got = _top_k(results, k=k, itr=itr)
    assert np.all(got == expected)