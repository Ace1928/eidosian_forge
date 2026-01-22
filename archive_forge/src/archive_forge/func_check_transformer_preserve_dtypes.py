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
def check_transformer_preserve_dtypes(name, transformer_orig):
    X, y = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]], random_state=0, cluster_std=0.1)
    X = StandardScaler().fit_transform(X)
    X = _enforce_estimator_tags_X(transformer_orig, X)
    for dtype in _safe_tags(transformer_orig, key='preserves_dtype'):
        X_cast = X.astype(dtype)
        transformer = clone(transformer_orig)
        set_random_state(transformer)
        X_trans1 = transformer.fit_transform(X_cast, y)
        X_trans2 = transformer.fit(X_cast, y).transform(X_cast)
        for Xt, method in zip([X_trans1, X_trans2], ['fit_transform', 'transform']):
            if isinstance(Xt, tuple):
                Xt = Xt[0]
            assert Xt.dtype == dtype, f'{name} (method={method}) does not preserve dtype. Original/Expected dtype={dtype.__name__}, got dtype={Xt.dtype}.'