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
def test_grid_search_cv_results_multimetric():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    n_splits = 3
    params = [dict(kernel=['rbf'], C=[1, 10], gamma=[0.1, 1]), dict(kernel=['poly'], degree=[1, 2])]
    grid_searches = []
    for scoring in ({'accuracy': make_scorer(accuracy_score), 'recall': make_scorer(recall_score)}, 'accuracy', 'recall'):
        grid_search = GridSearchCV(SVC(), cv=n_splits, param_grid=params, scoring=scoring, refit=False)
        grid_search.fit(X, y)
        grid_searches.append(grid_search)
    compare_cv_results_multimetric_with_single(*grid_searches)