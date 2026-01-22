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
def check_no_attributes_set_in_init(name, estimator_orig):
    """Check setting during init."""
    try:
        estimator = clone(estimator_orig)
    except AttributeError:
        raise AttributeError(f'Estimator {name} should store all parameters as an attribute during init.')
    if hasattr(type(estimator).__init__, 'deprecated_original'):
        return
    init_params = _get_args(type(estimator).__init__)
    if IS_PYPY:
        for key in ['obj']:
            if key in init_params:
                init_params.remove(key)
    parents_init_params = [param for params_parent in (_get_args(parent) for parent in type(estimator).__mro__) for param in params_parent]
    invalid_attr = set(vars(estimator)) - set(init_params) - set(parents_init_params)
    invalid_attr = set([attr for attr in invalid_attr if not attr.startswith('_')])
    assert not invalid_attr, 'Estimator %s should not set any attribute apart from parameters during init. Found attributes %s.' % (name, sorted(invalid_attr))