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
def test_stochastic_gradient_loss_param():
    param_grid = {'loss': ['log_loss']}
    X = np.arange(24).reshape(6, -1)
    y = [0, 0, 0, 1, 1, 1]
    clf = GridSearchCV(estimator=SGDClassifier(loss='hinge'), param_grid=param_grid, cv=3)
    assert not hasattr(clf, 'predict_proba')
    clf.fit(X, y)
    clf.predict_proba(X)
    clf.predict_log_proba(X)
    param_grid = {'loss': ['hinge']}
    clf = GridSearchCV(estimator=SGDClassifier(loss='hinge'), param_grid=param_grid, cv=3)
    assert not hasattr(clf, 'predict_proba')
    clf.fit(X, y)
    assert not hasattr(clf, 'predict_proba')