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
def check_set_output_transform(name, transformer_orig):
    tags = transformer_orig._get_tags()
    if '2darray' not in tags['X_types'] or tags['no_validation']:
        return
    rng = np.random.RandomState(0)
    transformer = clone(transformer_orig)
    X = rng.uniform(size=(20, 5))
    X = _enforce_estimator_tags_X(transformer_orig, X)
    y = rng.randint(0, 2, size=20)
    y = _enforce_estimator_tags_y(transformer_orig, y)
    set_random_state(transformer)

    def fit_then_transform(est):
        if name in CROSS_DECOMPOSITION:
            return est.fit(X, y).transform(X, y)
        return est.fit(X, y).transform(X)

    def fit_transform(est):
        return est.fit_transform(X, y)
    transform_methods = {'transform': fit_then_transform, 'fit_transform': fit_transform}
    for name, transform_method in transform_methods.items():
        transformer = clone(transformer)
        if not hasattr(transformer, name):
            continue
        X_trans_no_setting = transform_method(transformer)
        if name in CROSS_DECOMPOSITION:
            X_trans_no_setting = X_trans_no_setting[0]
        transformer.set_output(transform='default')
        X_trans_default = transform_method(transformer)
        if name in CROSS_DECOMPOSITION:
            X_trans_default = X_trans_default[0]
        assert_allclose_dense_sparse(X_trans_no_setting, X_trans_default)