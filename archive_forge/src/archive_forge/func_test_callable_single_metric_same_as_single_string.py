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
def test_callable_single_metric_same_as_single_string():

    def custom_scorer(est, X, y):
        y_pred = est.predict(X)
        return recall_score(y, y_pred)
    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    est = LinearSVC(dual='auto', random_state=42)
    search_callable = GridSearchCV(est, {'C': [0.1, 1]}, scoring=custom_scorer, refit=True)
    search_str = GridSearchCV(est, {'C': [0.1, 1]}, scoring='recall', refit='recall')
    search_list_str = GridSearchCV(est, {'C': [0.1, 1]}, scoring=['recall'], refit='recall')
    search_callable.fit(X, y)
    search_str.fit(X, y)
    search_list_str.fit(X, y)
    assert search_callable.best_score_ == pytest.approx(search_str.best_score_)
    assert search_callable.best_index_ == search_str.best_index_
    assert search_callable.score(X, y) == pytest.approx(search_str.score(X, y))
    assert search_list_str.best_score_ == pytest.approx(search_str.best_score_)
    assert search_list_str.best_index_ == search_str.best_index_
    assert search_list_str.score(X, y) == pytest.approx(search_str.score(X, y))