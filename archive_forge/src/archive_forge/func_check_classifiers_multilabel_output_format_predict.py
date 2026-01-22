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
def check_classifiers_multilabel_output_format_predict(name, classifier_orig):
    """Check the output of the `predict` method for classifiers supporting
    multilabel-indicator targets."""
    classifier = clone(classifier_orig)
    set_random_state(classifier)
    n_samples, test_size, n_outputs = (100, 25, 5)
    X, y = make_multilabel_classification(n_samples=n_samples, n_features=2, n_classes=n_outputs, n_labels=3, length=50, allow_unlabeled=True, random_state=0)
    X = scale(X)
    X_train, X_test = (X[:-test_size], X[-test_size:])
    y_train, y_test = (y[:-test_size], y[-test_size:])
    classifier.fit(X_train, y_train)
    response_method_name = 'predict'
    predict_method = getattr(classifier, response_method_name, None)
    if predict_method is None:
        raise SkipTest(f'{name} does not have a {response_method_name} method.')
    y_pred = predict_method(X_test)
    assert isinstance(y_pred, np.ndarray), f'{name}.predict is expected to output a NumPy array. Got {type(y_pred)} instead.'
    assert y_pred.shape == y_test.shape, f'{name}.predict outputs a NumPy array of shape {y_pred.shape} instead of {y_test.shape}.'
    assert y_pred.dtype == y_test.dtype, f'{name}.predict does not output the same dtype than the targets. Got {y_pred.dtype} instead of {y_test.dtype}.'