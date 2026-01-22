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
def test_classes__property():
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)
    Cs = [0.1, 1, 10]
    grid_search = GridSearchCV(LinearSVC(dual='auto', random_state=0), {'C': Cs})
    grid_search.fit(X, y)
    assert_array_equal(grid_search.best_estimator_.classes_, grid_search.classes_)
    grid_search = GridSearchCV(Ridge(), {'alpha': [1.0, 2.0]})
    grid_search.fit(X, y)
    assert not hasattr(grid_search, 'classes_')
    grid_search = GridSearchCV(LinearSVC(dual='auto', random_state=0), {'C': Cs})
    assert not hasattr(grid_search, 'classes_')
    grid_search = GridSearchCV(LinearSVC(dual='auto', random_state=0), {'C': Cs}, refit=False)
    grid_search.fit(X, y)
    assert not hasattr(grid_search, 'classes_')