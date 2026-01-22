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
def check_regressor_multioutput(name, estimator):
    estimator = clone(estimator)
    n_samples = n_features = 10
    if not _is_pairwise_metric(estimator):
        n_samples = n_samples + 1
    X, y = make_regression(random_state=42, n_targets=5, n_samples=n_samples, n_features=n_features)
    X = _enforce_estimator_tags_X(estimator, X)
    estimator.fit(X, y)
    y_pred = estimator.predict(X)
    assert y_pred.dtype == np.dtype('float64'), 'Multioutput predictions by a regressor are expected to be floating-point precision. Got {} instead'.format(y_pred.dtype)
    assert y_pred.shape == y.shape, 'The shape of the prediction for multioutput data is incorrect. Expected {}, got {}.'