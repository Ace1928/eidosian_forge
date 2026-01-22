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
def check_estimators_data_not_an_array(name, estimator_orig, X, y, obj_type):
    if name in CROSS_DECOMPOSITION:
        raise SkipTest('Skipping check_estimators_data_not_an_array for cross decomposition module as estimators are not deterministic.')
    estimator_1 = clone(estimator_orig)
    estimator_2 = clone(estimator_orig)
    set_random_state(estimator_1)
    set_random_state(estimator_2)
    if obj_type not in ['NotAnArray', 'PandasDataframe']:
        raise ValueError('Data type {0} not supported'.format(obj_type))
    if obj_type == 'NotAnArray':
        y_ = _NotAnArray(np.asarray(y))
        X_ = _NotAnArray(np.asarray(X))
    else:
        try:
            import pandas as pd
            y_ = np.asarray(y)
            if y_.ndim == 1:
                y_ = pd.Series(y_, copy=False)
            else:
                y_ = pd.DataFrame(y_, copy=False)
            X_ = pd.DataFrame(np.asarray(X), copy=False)
        except ImportError:
            raise SkipTest('pandas is not installed: not checking estimators for pandas objects.')
    estimator_1.fit(X_, y_)
    pred1 = estimator_1.predict(X_)
    estimator_2.fit(X, y)
    pred2 = estimator_2.predict(X)
    assert_allclose(pred1, pred2, atol=0.01, err_msg=name)