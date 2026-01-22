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
def check_param_validation(name, estimator_orig):
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(20, 5))
    y = rng.randint(0, 2, size=20)
    y = _enforce_estimator_tags_y(estimator_orig, y)
    estimator_params = estimator_orig.get_params(deep=False).keys()
    if estimator_params:
        validation_params = estimator_orig._parameter_constraints.keys()
        unexpected_params = set(validation_params) - set(estimator_params)
        missing_params = set(estimator_params) - set(validation_params)
        err_msg = f'Mismatch between _parameter_constraints and the parameters of {name}.\nConsider the unexpected parameters {unexpected_params} and expected but missing parameters {missing_params}'
        assert validation_params == estimator_params, err_msg
    param_with_bad_type = type('BadType', (), {})()
    fit_methods = ['fit', 'partial_fit', 'fit_transform', 'fit_predict']
    for param_name in estimator_params:
        constraints = estimator_orig._parameter_constraints[param_name]
        if constraints == 'no_validation':
            continue
        if any((isinstance(constraint, Interval) and constraint.type == Integral for constraint in constraints)) and any((isinstance(constraint, Interval) and constraint.type == Real for constraint in constraints)):
            raise ValueError(f"The constraint for parameter {param_name} of {name} can't have a mix of intervals of Integral and Real types. Use the type RealNotInt instead of Real.")
        match = f"The '{param_name}' parameter of {name} must be .* Got .* instead."
        err_msg = f'{name} does not raise an informative error message when the parameter {param_name} does not have a valid type or value.'
        estimator = clone(estimator_orig)
        estimator.set_params(**{param_name: param_with_bad_type})
        for method in fit_methods:
            if not hasattr(estimator, method):
                continue
            err_msg = f"{name} does not raise an informative error message when the parameter {param_name} does not have a valid type. If any Python type is valid, the constraint should be 'no_validation'."
            with raises(InvalidParameterError, match=match, err_msg=err_msg):
                if any((isinstance(X_type, str) and X_type.endswith('labels') for X_type in _safe_tags(estimator, key='X_types'))):
                    getattr(estimator, method)(y)
                else:
                    getattr(estimator, method)(X, y)
        constraints = [make_constraint(constraint) for constraint in constraints]
        for constraint in constraints:
            try:
                bad_value = generate_invalid_param_val(constraint)
            except NotImplementedError:
                continue
            estimator.set_params(**{param_name: bad_value})
            for method in fit_methods:
                if not hasattr(estimator, method):
                    continue
                err_msg = f"{name} does not raise an informative error message when the parameter {param_name} does not have a valid value.\nConstraints should be disjoint. For instance [StrOptions({{'a_string'}}), str] is not a acceptable set of constraint because generating an invalid string for the first constraint will always produce a valid string for the second constraint."
                with raises(InvalidParameterError, match=match, err_msg=err_msg):
                    if any((X_type.endswith('labels') for X_type in _safe_tags(estimator, key='X_types'))):
                        getattr(estimator, method)(y)
                    else:
                        getattr(estimator, method)(X, y)