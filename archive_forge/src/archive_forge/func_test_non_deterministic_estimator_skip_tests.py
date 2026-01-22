import importlib
import sys
import unittest
import warnings
from numbers import Integral, Real
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn import config_context, get_config
from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_multilabel_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning, SkipTestWarning
from sklearn.linear_model import (
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC, NuSVC
from sklearn.utils import _array_api, all_estimators, deprecated
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
def test_non_deterministic_estimator_skip_tests():
    for est in [MinimalTransformer, MinimalRegressor, MinimalClassifier]:
        all_tests = list(_yield_all_checks(est()))
        assert check_methods_sample_order_invariance in all_tests
        assert check_methods_subset_invariance in all_tests

        class Estimator(est):

            def _more_tags(self):
                return {'non_deterministic': True}
        all_tests = list(_yield_all_checks(Estimator()))
        assert check_methods_sample_order_invariance not in all_tests
        assert check_methods_subset_invariance not in all_tests