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
@pytest.mark.parametrize('search_cv', [RandomizedSearchCV(estimator=LocalOutlierFactor(novelty=True), param_distributions={'n_neighbors': [5, 10]}, scoring='precision'), GridSearchCV(estimator=LocalOutlierFactor(novelty=True), param_grid={'n_neighbors': [5, 10]}, scoring='precision')])
def test_search_cv_score_samples_method(search_cv):
    rng = np.random.RandomState(42)
    n_samples = 300
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers
    X = make_blobs(n_samples=n_inliers, n_features=2, centers=[[0, 0], [0, 0]], cluster_std=0.5, random_state=0)[0]
    X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)
    y_true = np.array([1] * n_samples)
    y_true[-n_outliers:] = -1
    search_cv.fit(X, y_true)
    assert_allclose(search_cv.score_samples(X), search_cv.best_estimator_.score_samples(X))