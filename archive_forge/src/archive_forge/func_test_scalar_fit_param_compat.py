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
@pytest.mark.parametrize('SearchCV, param_search', [(GridSearchCV, {'alpha': [0.1, 0.01]}), (RandomizedSearchCV, {'alpha': uniform(0.01, 0.1)})])
def test_scalar_fit_param_compat(SearchCV, param_search):
    X_train, X_valid, y_train, y_valid = train_test_split(*make_classification(random_state=42), random_state=42)

    class _FitParamClassifier(SGDClassifier):

        def fit(self, X, y, sample_weight=None, tuple_of_arrays=None, scalar_param=None, callable_param=None):
            super().fit(X, y, sample_weight=sample_weight)
            assert scalar_param > 0
            assert callable(callable_param)
            assert isinstance(tuple_of_arrays, tuple)
            assert tuple_of_arrays[0].ndim == 2
            assert tuple_of_arrays[1].ndim == 1
            return self

    def _fit_param_callable():
        pass
    model = SearchCV(_FitParamClassifier(), param_search)
    fit_params = {'tuple_of_arrays': (X_valid, y_valid), 'callable_param': _fit_param_callable, 'scalar_param': 42}
    model.fit(X_train, y_train, **fit_params)