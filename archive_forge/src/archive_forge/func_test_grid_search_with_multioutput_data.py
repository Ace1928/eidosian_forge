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
def test_grid_search_with_multioutput_data():
    X, y = make_multilabel_classification(return_indicator=True, random_state=0)
    est_parameters = {'max_depth': [1, 2, 3, 4]}
    cv = KFold()
    estimators = [DecisionTreeRegressor(random_state=0), DecisionTreeClassifier(random_state=0)]
    for est in estimators:
        grid_search = GridSearchCV(est, est_parameters, cv=cv)
        grid_search.fit(X, y)
        res_params = grid_search.cv_results_['params']
        for cand_i in range(len(res_params)):
            est.set_params(**res_params[cand_i])
            for i, (train, test) in enumerate(cv.split(X, y)):
                est.fit(X[train], y[train])
                correct_score = est.score(X[test], y[test])
                assert_almost_equal(correct_score, grid_search.cv_results_['split%d_test_score' % i][cand_i])
    for est in estimators:
        random_search = RandomizedSearchCV(est, est_parameters, cv=cv, n_iter=3)
        random_search.fit(X, y)
        res_params = random_search.cv_results_['params']
        for cand_i in range(len(res_params)):
            est.set_params(**res_params[cand_i])
            for i, (train, test) in enumerate(cv.split(X, y)):
                est.fit(X[train], y[train])
                correct_score = est.score(X[test], y[test])
                assert_almost_equal(correct_score, random_search.cv_results_['split%d_test_score' % i][cand_i])