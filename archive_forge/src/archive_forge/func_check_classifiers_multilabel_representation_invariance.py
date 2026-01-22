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
def check_classifiers_multilabel_representation_invariance(name, classifier_orig):
    X, y = make_multilabel_classification(n_samples=100, n_features=2, n_classes=5, n_labels=3, length=50, allow_unlabeled=True, random_state=0)
    X = scale(X)
    X_train, y_train = (X[:80], y[:80])
    X_test = X[80:]
    y_train_list_of_lists = y_train.tolist()
    y_train_list_of_arrays = list(y_train)
    classifier = clone(classifier_orig)
    set_random_state(classifier)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    y_pred_list_of_lists = classifier.fit(X_train, y_train_list_of_lists).predict(X_test)
    y_pred_list_of_arrays = classifier.fit(X_train, y_train_list_of_arrays).predict(X_test)
    assert_array_equal(y_pred, y_pred_list_of_arrays)
    assert_array_equal(y_pred, y_pred_list_of_lists)
    assert y_pred.dtype == y_pred_list_of_arrays.dtype
    assert y_pred.dtype == y_pred_list_of_lists.dtype
    assert type(y_pred) == type(y_pred_list_of_arrays)
    assert type(y_pred) == type(y_pred_list_of_lists)