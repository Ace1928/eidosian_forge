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
def test_parameter_grid():
    params1 = {'foo': [1, 2, 3]}
    grid1 = ParameterGrid(params1)
    assert isinstance(grid1, Iterable)
    assert isinstance(grid1, Sized)
    assert len(grid1) == 3
    assert_grid_iter_equals_getitem(grid1)
    params2 = {'foo': [4, 2], 'bar': ['ham', 'spam', 'eggs']}
    grid2 = ParameterGrid(params2)
    assert len(grid2) == 6
    for i in range(2):
        points = set((tuple(chain(*sorted(p.items()))) for p in grid2))
        assert points == set((('bar', x, 'foo', y) for x, y in product(params2['bar'], params2['foo'])))
    assert_grid_iter_equals_getitem(grid2)
    empty = ParameterGrid({})
    assert len(empty) == 1
    assert list(empty) == [{}]
    assert_grid_iter_equals_getitem(empty)
    with pytest.raises(IndexError):
        empty[1]
    has_empty = ParameterGrid([{'C': [1, 10]}, {}, {'C': [0.5]}])
    assert len(has_empty) == 4
    assert list(has_empty) == [{'C': 1}, {'C': 10}, {}, {'C': 0.5}]
    assert_grid_iter_equals_getitem(has_empty)