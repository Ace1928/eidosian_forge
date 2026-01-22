import pickle
import re
import warnings
from contextlib import nullcontext
from copy import deepcopy
from functools import partial, wraps
from inspect import signature
from numbers import Integral, Real
import joblib
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from .. import config_context
from ..base import (
from ..datasets import (
from ..exceptions import DataConversionWarning, NotFittedError, SkipTestWarning
from ..feature_selection import SelectFromModel, SelectKBest
from ..linear_model import (
from ..metrics import accuracy_score, adjusted_rand_score, f1_score
from ..metrics.pairwise import linear_kernel, pairwise_distances, rbf_kernel
from ..model_selection import ShuffleSplit, train_test_split
from ..model_selection._validation import _safe_split
from ..pipeline import make_pipeline
from ..preprocessing import StandardScaler, scale
from ..random_projection import BaseRandomProjection
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils._array_api import (
from ..utils._array_api import (
from ..utils._param_validation import (
from ..utils.fixes import parse_version, sp_version
from ..utils.validation import check_is_fitted
from . import IS_PYPY, is_scalar_nan, shuffle
from ._param_validation import Interval
from ._tags import (
from ._testing import (
from .validation import _num_samples, has_fit_parameter
@ignore_warnings
def check_estimators_pickle(name, estimator_orig, readonly_memmap=False):
    """Test that we can pickle all estimators."""
    check_methods = ['predict', 'transform', 'decision_function', 'predict_proba']
    X, y = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]], random_state=0, n_features=2, cluster_std=0.1)
    X = _enforce_estimator_tags_X(estimator_orig, X, kernel=rbf_kernel)
    tags = _safe_tags(estimator_orig)
    if tags['allow_nan']:
        rng = np.random.RandomState(42)
        mask = rng.choice(X.size, 10, replace=False)
        X.reshape(-1)[mask] = np.nan
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator)
    estimator.fit(X, y)
    if readonly_memmap:
        unpickled_estimator = create_memmap_backed_data(estimator)
    else:
        pickled_estimator = pickle.dumps(estimator)
        module_name = estimator.__module__
        if module_name.startswith('sklearn.') and (not ('test_' in module_name or module_name.endswith('_testing'))):
            assert b'_sklearn_version' in pickled_estimator
        unpickled_estimator = pickle.loads(pickled_estimator)
    result = dict()
    for method in check_methods:
        if hasattr(estimator, method):
            result[method] = getattr(estimator, method)(X)
    for method in result:
        unpickled_result = getattr(unpickled_estimator, method)(X)
        assert_allclose_dense_sparse(result[method], unpickled_result)