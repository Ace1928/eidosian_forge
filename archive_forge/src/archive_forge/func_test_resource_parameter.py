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
def test_resource_parameter(Est):
    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {'a': [1, 2], 'b': list(range(10))}
    base_estimator = FastClassifier()
    sh = Est(base_estimator, param_grid, cv=2, resource='c', max_resources=10, factor=3)
    sh.fit(X, y)
    assert set(sh.n_resources_) == set([1, 3, 9])
    for r_i, params, param_c in zip(sh.cv_results_['n_resources'], sh.cv_results_['params'], sh.cv_results_['param_c']):
        assert r_i == params['c'] == param_c
    with pytest.raises(ValueError, match='Cannot use resource=1234 which is not supported '):
        sh = HalvingGridSearchCV(base_estimator, param_grid, cv=2, resource='1234', max_resources=10)
        sh.fit(X, y)
    with pytest.raises(ValueError, match='Cannot use parameter c as the resource since it is part of the searched parameters.'):
        param_grid = {'a': [1, 2], 'b': [1, 2], 'c': [1, 3]}
        sh = HalvingGridSearchCV(base_estimator, param_grid, cv=2, resource='c', max_resources=10)
        sh.fit(X, y)