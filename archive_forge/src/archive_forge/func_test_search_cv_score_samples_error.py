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
@pytest.mark.parametrize('search_cv', [RandomizedSearchCV(estimator=DecisionTreeClassifier(), param_distributions={'max_depth': [5, 10]}), GridSearchCV(estimator=DecisionTreeClassifier(), param_grid={'max_depth': [5, 10]})])
def test_search_cv_score_samples_error(search_cv):
    X, y = make_blobs(n_samples=100, n_features=4, random_state=42)
    search_cv.fit(X, y)
    outer_msg = f"'{search_cv.__class__.__name__}' has no attribute 'score_samples'"
    inner_msg = "'DecisionTreeClassifier' object has no attribute 'score_samples'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        search_cv.score_samples(X)
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg == str(exec_info.value.__cause__)