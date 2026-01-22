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
def check_fit_check_is_fitted(name, estimator_orig):
    rng = np.random.RandomState(42)
    estimator = clone(estimator_orig)
    set_random_state(estimator)
    if 'warm_start' in estimator.get_params():
        estimator.set_params(warm_start=False)
    n_samples = 100
    X = rng.normal(loc=100, size=(n_samples, 2))
    X = _enforce_estimator_tags_X(estimator, X)
    if is_regressor(estimator_orig):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)
    y = _enforce_estimator_tags_y(estimator, y)
    if not _safe_tags(estimator).get('stateless', False):
        try:
            check_is_fitted(estimator)
            raise AssertionError(f'{estimator.__class__.__name__} passes check_is_fitted before being fit!')
        except NotFittedError:
            pass
    estimator.fit(X, y)
    try:
        check_is_fitted(estimator)
    except NotFittedError as e:
        raise NotFittedError('Estimator fails to pass `check_is_fitted` even though it has been fit.') from e