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
def test_grid_search_groups():
    rng = np.random.RandomState(0)
    X, y = make_classification(n_samples=15, n_classes=2, random_state=0)
    groups = rng.randint(0, 3, 15)
    clf = LinearSVC(dual='auto', random_state=0)
    grid = {'C': [1]}
    group_cvs = [LeaveOneGroupOut(), LeavePGroupsOut(2), GroupKFold(n_splits=3), GroupShuffleSplit()]
    error_msg = "The 'groups' parameter should not be None."
    for cv in group_cvs:
        gs = GridSearchCV(clf, grid, cv=cv)
        with pytest.raises(ValueError, match=error_msg):
            gs.fit(X, y)
        gs.fit(X, y, groups=groups)
    non_group_cvs = [StratifiedKFold(), StratifiedShuffleSplit()]
    for cv in non_group_cvs:
        gs = GridSearchCV(clf, grid, cv=cv)
        gs.fit(X, y)