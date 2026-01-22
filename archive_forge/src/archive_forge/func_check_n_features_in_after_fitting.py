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
@ignore_warnings(category=FutureWarning)
def check_n_features_in_after_fitting(name, estimator_orig):
    tags = _safe_tags(estimator_orig)
    is_supported_X_types = '2darray' in tags['X_types'] or 'categorical' in tags['X_types']
    if not is_supported_X_types or tags['no_validation']:
        return
    rng = np.random.RandomState(0)
    estimator = clone(estimator_orig)
    set_random_state(estimator)
    if 'warm_start' in estimator.get_params():
        estimator.set_params(warm_start=False)
    n_samples = 150
    X = rng.normal(size=(n_samples, 8))
    X = _enforce_estimator_tags_X(estimator, X)
    if is_regressor(estimator):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)
    y = _enforce_estimator_tags_y(estimator, y)
    estimator.fit(X, y)
    assert estimator.n_features_in_ == X.shape[1]
    check_methods = ['predict', 'transform', 'decision_function', 'predict_proba', 'score']
    X_bad = X[:, [1]]
    msg = f'X has 1 features, but \\w+ is expecting {X.shape[1]} features as input'
    for method in check_methods:
        if not hasattr(estimator, method):
            continue
        callable_method = getattr(estimator, method)
        if method == 'score':
            callable_method = partial(callable_method, y=y)
        with raises(ValueError, match=msg):
            callable_method(X_bad)
    if not hasattr(estimator, 'partial_fit'):
        return
    estimator = clone(estimator_orig)
    if is_classifier(estimator):
        estimator.partial_fit(X, y, classes=np.unique(y))
    else:
        estimator.partial_fit(X, y)
    assert estimator.n_features_in_ == X.shape[1]
    with raises(ValueError, match=msg):
        estimator.partial_fit(X_bad, y)