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
@pytest.mark.parametrize('params, expected_error_message', [({'n_candidates': 'exhaust', 'min_resources': 'exhaust'}, "cannot be both set to 'exhaust'")])
def test_input_errors_randomized(params, expected_error_message):
    base_estimator = FastClassifier()
    param_grid = {'a': [1]}
    X, y = make_classification(100)
    sh = HalvingRandomSearchCV(base_estimator, param_grid, **params)
    with pytest.raises(ValueError, match=expected_error_message):
        sh.fit(X, y)