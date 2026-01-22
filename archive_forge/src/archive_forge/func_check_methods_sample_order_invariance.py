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
def check_methods_sample_order_invariance(name, estimator_orig):
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = X[:, 0].astype(np.int64)
    if _safe_tags(estimator_orig, key='binary_only'):
        y[y == 2] = 1
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    if hasattr(estimator, 'n_components'):
        estimator.n_components = 1
    if hasattr(estimator, 'n_clusters'):
        estimator.n_clusters = 2
    set_random_state(estimator, 1)
    estimator.fit(X, y)
    idx = np.random.permutation(X.shape[0])
    for method in ['predict', 'transform', 'decision_function', 'score_samples', 'predict_proba']:
        msg = '{method} of {name} is not invariant when applied to a datasetwith different sample order.'.format(method=method, name=name)
        if hasattr(estimator, method):
            assert_allclose_dense_sparse(getattr(estimator, method)(X)[idx], getattr(estimator, method)(X[idx]), atol=1e-09, err_msg=msg)