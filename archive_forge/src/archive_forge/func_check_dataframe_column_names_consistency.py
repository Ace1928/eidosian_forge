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
def check_dataframe_column_names_consistency(name, estimator_orig):
    try:
        import pandas as pd
    except ImportError:
        raise SkipTest('pandas is not installed: not checking column name consistency for pandas')
    tags = _safe_tags(estimator_orig)
    is_supported_X_types = '2darray' in tags['X_types'] or 'categorical' in tags['X_types']
    if not is_supported_X_types or tags['no_validation']:
        return
    rng = np.random.RandomState(0)
    estimator = clone(estimator_orig)
    set_random_state(estimator)
    X_orig = rng.normal(size=(150, 8))
    X_orig = _enforce_estimator_tags_X(estimator, X_orig)
    n_samples, n_features = X_orig.shape
    names = np.array([f'col_{i}' for i in range(n_features)])
    X = pd.DataFrame(X_orig, columns=names, copy=False)
    if is_regressor(estimator):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)
    y = _enforce_estimator_tags_y(estimator, y)
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='X does not have valid feature names', category=UserWarning, module='sklearn')
        estimator.fit(X, y)
    if not hasattr(estimator, 'feature_names_in_'):
        raise ValueError('Estimator does not have a feature_names_in_ attribute after fitting with a dataframe')
    assert isinstance(estimator.feature_names_in_, np.ndarray)
    assert estimator.feature_names_in_.dtype == object
    assert_array_equal(estimator.feature_names_in_, names)
    module_name = estimator_orig.__module__
    if module_name.startswith('sklearn.') and (not ('test_' in module_name or module_name.endswith('_testing'))) and ('feature_names_in_' not in estimator_orig.__doc__):
        raise ValueError(f'Estimator {name} does not document its feature_names_in_ attribute')
    check_methods = []
    for method in ('predict', 'transform', 'decision_function', 'predict_proba', 'score', 'score_samples', 'predict_log_proba'):
        if not hasattr(estimator, method):
            continue
        callable_method = getattr(estimator, method)
        if method == 'score':
            callable_method = partial(callable_method, y=y)
        check_methods.append((method, callable_method))
    for _, method in check_methods:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='X does not have valid feature names', category=UserWarning, module='sklearn')
            method(X)
    invalid_names = [(names[::-1], 'Feature names must be in the same order as they were in fit.'), ([f'another_prefix_{i}' for i in range(n_features)], 'Feature names unseen at fit time:\n- another_prefix_0\n- another_prefix_1\n'), (names[:3], f'Feature names seen at fit time, yet now missing:\n- {min(names[3:])}\n')]
    params = {key: value for key, value in estimator.get_params().items() if 'early_stopping' in key}
    early_stopping_enabled = any((value is True for value in params.values()))
    for invalid_name, additional_message in invalid_names:
        X_bad = pd.DataFrame(X, columns=invalid_name, copy=False)
        expected_msg = re.escape(f'The feature names should match those that were passed during fit.\n{additional_message}')
        for name, method in check_methods:
            with raises(ValueError, match=expected_msg, err_msg=f'{name} did not raise'):
                method(X_bad)
        if not hasattr(estimator, 'partial_fit') or early_stopping_enabled:
            continue
        estimator = clone(estimator_orig)
        if is_classifier(estimator):
            classes = np.unique(y)
            estimator.partial_fit(X, y, classes=classes)
        else:
            estimator.partial_fit(X, y)
        with raises(ValueError, match=expected_msg):
            estimator.partial_fit(X_bad, y)