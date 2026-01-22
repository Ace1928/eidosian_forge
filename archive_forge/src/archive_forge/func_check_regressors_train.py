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
def check_regressors_train(name, regressor_orig, readonly_memmap=False, X_dtype=np.float64):
    X, y = _regression_dataset()
    X = X.astype(X_dtype)
    y = scale(y)
    regressor = clone(regressor_orig)
    X = _enforce_estimator_tags_X(regressor, X)
    y = _enforce_estimator_tags_y(regressor, y)
    if name in CROSS_DECOMPOSITION:
        rnd = np.random.RandomState(0)
        y_ = np.vstack([y, 2 * y + rnd.randint(2, size=len(y))])
        y_ = y_.T
    else:
        y_ = y
    if readonly_memmap:
        X, y, y_ = create_memmap_backed_data([X, y, y_])
    if not hasattr(regressor, 'alphas') and hasattr(regressor, 'alpha'):
        regressor.alpha = 0.01
    if name == 'PassiveAggressiveRegressor':
        regressor.C = 0.01
    with raises(ValueError, err_msg=f'The classifier {name} does not raise an error when incorrect/malformed input data for fit is passed. The number of training examples is not the same as the number of labels. Perhaps use check_X_y in fit.'):
        regressor.fit(X, y[:-1])
    set_random_state(regressor)
    regressor.fit(X, y_)
    regressor.fit(X.tolist(), y_.tolist())
    y_pred = regressor.predict(X)
    assert y_pred.shape == y_.shape
    if not _safe_tags(regressor, key='poor_score'):
        assert regressor.score(X, y_) > 0.5