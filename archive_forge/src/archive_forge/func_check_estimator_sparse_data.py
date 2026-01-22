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
def check_estimator_sparse_data(name, estimator_orig):
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(40, 3))
    X[X < 0.8] = 0
    X = _enforce_estimator_tags_X(estimator_orig, X)
    X_csr = sparse.csr_matrix(X)
    y = (4 * rng.uniform(size=40)).astype(int)
    with ignore_warnings(category=FutureWarning):
        estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    tags = _safe_tags(estimator_orig)
    for matrix_format, X in _generate_sparse_matrix(X_csr):
        with ignore_warnings(category=FutureWarning):
            estimator = clone(estimator_orig)
            if name in ['Scaler', 'StandardScaler']:
                estimator.set_params(with_mean=False)
        if '64' in matrix_format:
            err_msg = f"Estimator {name} doesn't seem to support {matrix_format} matrix, and is not failing gracefully, e.g. by using check_array(X, accept_large_sparse=False)"
        else:
            err_msg = f"Estimator {name} doesn't seem to fail gracefully on sparse data: error message should state explicitly that sparse input is not supported if this is not the case."
        with raises((TypeError, ValueError), match=['sparse', 'Sparse'], may_pass=True, err_msg=err_msg):
            with ignore_warnings(category=FutureWarning):
                estimator.fit(X, y)
            if hasattr(estimator, 'predict'):
                pred = estimator.predict(X)
                if tags['multioutput_only']:
                    assert pred.shape == (X.shape[0], 1)
                else:
                    assert pred.shape == (X.shape[0],)
            if hasattr(estimator, 'predict_proba'):
                probs = estimator.predict_proba(X)
                if tags['binary_only']:
                    expected_probs_shape = (X.shape[0], 2)
                else:
                    expected_probs_shape = (X.shape[0], 4)
                assert probs.shape == expected_probs_shape