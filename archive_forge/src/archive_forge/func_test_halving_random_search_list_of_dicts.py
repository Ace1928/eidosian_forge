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
def test_halving_random_search_list_of_dicts():
    """Check the behaviour of the `HalvingRandomSearchCV` with `param_distribution`
    being a list of dictionary.
    """
    X, y = make_classification(n_samples=150, n_features=4, random_state=42)
    params = [{'kernel': ['rbf'], 'C': expon(scale=10), 'gamma': expon(scale=0.1)}, {'kernel': ['poly'], 'degree': [2, 3]}]
    param_keys = ('param_C', 'param_degree', 'param_gamma', 'param_kernel')
    score_keys = ('mean_test_score', 'mean_train_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'split0_train_score', 'split1_train_score', 'split2_train_score', 'std_test_score', 'std_train_score', 'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time')
    extra_keys = ('n_resources', 'iter')
    search = HalvingRandomSearchCV(SVC(), cv=3, param_distributions=params, return_train_score=True, random_state=0)
    search.fit(X, y)
    n_candidates = sum(search.n_candidates_)
    cv_results = search.cv_results_
    check_cv_results_keys(cv_results, param_keys, score_keys, n_candidates, extra_keys)
    check_cv_results_array_types(search, param_keys, score_keys)
    assert all((cv_results['param_C'].mask[i] and cv_results['param_gamma'].mask[i] and (not cv_results['param_degree'].mask[i]) for i in range(n_candidates) if cv_results['param_kernel'][i] == 'poly'))
    assert all((not cv_results['param_C'].mask[i] and (not cv_results['param_gamma'].mask[i]) and cv_results['param_degree'].mask[i] for i in range(n_candidates) if cv_results['param_kernel'][i] == 'rbf'))