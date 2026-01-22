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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_grid_search_sparse_scoring(csr_container):
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)
    clf = LinearSVC(dual='auto')
    cv = GridSearchCV(clf, {'C': [0.1, 1.0]}, scoring='f1')
    cv.fit(X_[:180], y_[:180])
    y_pred = cv.predict(X_[180:])
    C = cv.best_estimator_.C
    X_ = csr_container(X_)
    clf = LinearSVC(dual='auto')
    cv = GridSearchCV(clf, {'C': [0.1, 1.0]}, scoring='f1')
    cv.fit(X_[:180], y_[:180])
    y_pred2 = cv.predict(X_[180:])
    C2 = cv.best_estimator_.C
    assert_array_equal(y_pred, y_pred2)
    assert C == C2

    def f1_loss(y_true_, y_pred_):
        return -f1_score(y_true_, y_pred_)
    F1Loss = make_scorer(f1_loss, greater_is_better=False)
    cv = GridSearchCV(clf, {'C': [0.1, 1.0]}, scoring=F1Loss)
    cv.fit(X_[:180], y_[:180])
    y_pred3 = cv.predict(X_[180:])
    C3 = cv.best_estimator_.C
    assert C == C3
    assert_array_equal(y_pred, y_pred3)