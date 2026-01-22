import pickle
import re
import sys
from collections.abc import Iterable, Sized
from functools import partial
from io import StringIO
from itertools import chain, product
from types import GeneratorType
import numpy as np
import pytest
from scipy.stats import bernoulli, expon, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import (
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import (
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection.tests.common import OneTimeSplitter
from sklearn.neighbors import KernelDensity, KNeighborsClassifier, LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
def test_random_search_cv_results():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    n_search_iter = 30
    params = [{'kernel': ['rbf'], 'C': expon(scale=10), 'gamma': expon(scale=0.1)}, {'kernel': ['poly'], 'degree': [2, 3]}]
    param_keys = ('param_C', 'param_degree', 'param_gamma', 'param_kernel')
    score_keys = ('mean_test_score', 'mean_train_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'split0_train_score', 'split1_train_score', 'split2_train_score', 'std_test_score', 'std_train_score', 'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time')
    n_candidates = n_search_iter
    search = RandomizedSearchCV(SVC(), n_iter=n_search_iter, cv=3, param_distributions=params, return_train_score=True)
    search.fit(X, y)
    cv_results = search.cv_results_
    check_cv_results_array_types(search, param_keys, score_keys)
    check_cv_results_keys(cv_results, param_keys, score_keys, n_candidates)
    assert all((cv_results['param_C'].mask[i] and cv_results['param_gamma'].mask[i] and (not cv_results['param_degree'].mask[i]) for i in range(n_candidates) if cv_results['param_kernel'][i] == 'poly'))
    assert all((not cv_results['param_C'].mask[i] and (not cv_results['param_gamma'].mask[i]) and cv_results['param_degree'].mask[i] for i in range(n_candidates) if cv_results['param_kernel'][i] == 'rbf'))