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
def test_grid_search_bad_param_grid():
    X, y = make_classification(n_samples=10, n_features=5, random_state=0)
    param_dict = {'C': 1}
    clf = SVC(gamma='auto')
    error_msg = re.escape("Parameter grid for parameter 'C' needs to be a list or a numpy array, but got 1 (of type int) instead. Single values need to be wrapped in a list with one element.")
    search = GridSearchCV(clf, param_dict)
    with pytest.raises(TypeError, match=error_msg):
        search.fit(X, y)
    param_dict = {'C': []}
    clf = SVC()
    error_msg = re.escape("Parameter grid for parameter 'C' need to be a non-empty sequence, got: []")
    search = GridSearchCV(clf, param_dict)
    with pytest.raises(ValueError, match=error_msg):
        search.fit(X, y)
    param_dict = {'C': '1,2,3'}
    clf = SVC(gamma='auto')
    error_msg = re.escape("Parameter grid for parameter 'C' needs to be a list or a numpy array, but got '1,2,3' (of type str) instead. Single values need to be wrapped in a list with one element.")
    search = GridSearchCV(clf, param_dict)
    with pytest.raises(TypeError, match=error_msg):
        search.fit(X, y)
    param_dict = {'C': np.ones((3, 2))}
    clf = SVC()
    search = GridSearchCV(clf, param_dict)
    with pytest.raises(ValueError):
        search.fit(X, y)