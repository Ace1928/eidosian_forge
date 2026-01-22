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
@pytest.mark.parametrize('pairwise', [True, False])
def test_search_cv_pairwise_property_delegated_to_base_estimator(pairwise):
    """
    Test implementation of BaseSearchCV has the pairwise tag
    which matches the pairwise tag of its estimator.
    This test make sure pairwise tag is delegated to the base estimator.

    Non-regression test for issue #13920.
    """

    class TestEstimator(BaseEstimator):

        def _more_tags(self):
            return {'pairwise': pairwise}
    est = TestEstimator()
    attr_message = 'BaseSearchCV pairwise tag must match estimator'
    cv = GridSearchCV(est, {'n_neighbors': [10]})
    assert pairwise == cv._get_tags()['pairwise'], attr_message