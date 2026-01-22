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
def check_estimators_nan_inf(name, estimator_orig):
    rnd = np.random.RandomState(0)
    X_train_finite = _enforce_estimator_tags_X(estimator_orig, rnd.uniform(size=(10, 3)))
    X_train_nan = rnd.uniform(size=(10, 3))
    X_train_nan[0, 0] = np.nan
    X_train_inf = rnd.uniform(size=(10, 3))
    X_train_inf[0, 0] = np.inf
    y = np.ones(10)
    y[:5] = 0
    y = _enforce_estimator_tags_y(estimator_orig, y)
    error_string_fit = f"Estimator {name} doesn't check for NaN and inf in fit."
    error_string_predict = f"Estimator {name} doesn't check for NaN and inf in predict."
    error_string_transform = f"Estimator {name} doesn't check for NaN and inf in transform."
    for X_train in [X_train_nan, X_train_inf]:
        with ignore_warnings(category=FutureWarning):
            estimator = clone(estimator_orig)
            set_random_state(estimator, 1)
            with raises(ValueError, match=['inf', 'NaN'], err_msg=error_string_fit):
                estimator.fit(X_train, y)
            estimator.fit(X_train_finite, y)
            if hasattr(estimator, 'predict'):
                with raises(ValueError, match=['inf', 'NaN'], err_msg=error_string_predict):
                    estimator.predict(X_train)
            if hasattr(estimator, 'transform'):
                with raises(ValueError, match=['inf', 'NaN'], err_msg=error_string_transform):
                    estimator.transform(X_train)