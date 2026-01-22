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
@pytest.mark.parametrize('Est', (HalvingRandomSearchCV, HalvingGridSearchCV))
@pytest.mark.parametrize('max_resources, n_iterations, n_possible_iterations', [('auto', 5, 9), (1024, 5, 9), (700, 5, 8), (512, 5, 8), (511, 5, 7), (32, 4, 4), (31, 3, 3), (16, 3, 3), (4, 1, 1)])
def test_n_iterations(Est, max_resources, n_iterations, n_possible_iterations):
    n_samples = 1024
    X, y = make_classification(n_samples=n_samples, random_state=1)
    param_grid = {'a': [1, 2], 'b': list(range(10))}
    base_estimator = FastClassifier()
    factor = 2
    sh = Est(base_estimator, param_grid, cv=2, factor=factor, max_resources=max_resources, min_resources=4)
    if Est is HalvingRandomSearchCV:
        sh.set_params(n_candidates=20)
    sh.fit(X, y)
    assert sh.n_required_iterations_ == 5
    assert sh.n_iterations_ == n_iterations
    assert sh.n_possible_iterations_ == n_possible_iterations