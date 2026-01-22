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
@ignore_warnings
def test_pandas_input():
    types = [(MockDataFrame, MockDataFrame)]
    try:
        from pandas import DataFrame, Series
        types.append((DataFrame, Series))
    except ImportError:
        pass
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)
    for InputFeatureType, TargetType in types:
        X_df, y_ser = (InputFeatureType(X), TargetType(y))

        def check_df(x):
            return isinstance(x, InputFeatureType)

        def check_series(x):
            return isinstance(x, TargetType)
        clf = CheckingClassifier(check_X=check_df, check_y=check_series)
        grid_search = GridSearchCV(clf, {'foo_param': [1, 2, 3]})
        grid_search.fit(X_df, y_ser).score(X_df, y_ser)
        grid_search.predict(X_df)
        assert hasattr(grid_search, 'cv_results_')