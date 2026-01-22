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
def test_callable_multimetric_clf_all_fits_fail():

    def custom_scorer(est, X, y):
        return {'acc': 1}
    X, y = make_classification(n_samples=20, n_features=10, random_state=0)
    clf = FailingClassifier()
    gs = GridSearchCV(clf, [{'parameter': [FailingClassifier.FAILING_PARAMETER] * 3}], scoring=custom_scorer, refit=False, error_score=0.1)
    individual_fit_error_message = 'ValueError: Failing classifier failed as required'
    error_message = re.compile(f'All the 15 fits failed.+your model is misconfigured.+{individual_fit_error_message}', flags=re.DOTALL)
    with pytest.raises(ValueError, match=error_message):
        gs.fit(X, y)