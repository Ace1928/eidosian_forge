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
def check_classifiers_classes(name, classifier_orig):
    X_multiclass, y_multiclass = make_blobs(n_samples=30, random_state=0, cluster_std=0.1)
    X_multiclass, y_multiclass = shuffle(X_multiclass, y_multiclass, random_state=7)
    X_multiclass = StandardScaler().fit_transform(X_multiclass)
    X_binary = X_multiclass[y_multiclass != 2]
    y_binary = y_multiclass[y_multiclass != 2]
    X_multiclass = _enforce_estimator_tags_X(classifier_orig, X_multiclass)
    X_binary = _enforce_estimator_tags_X(classifier_orig, X_binary)
    labels_multiclass = ['one', 'two', 'three']
    labels_binary = ['one', 'two']
    y_names_multiclass = np.take(labels_multiclass, y_multiclass)
    y_names_binary = np.take(labels_binary, y_binary)
    problems = [(X_binary, y_binary, y_names_binary)]
    if not _safe_tags(classifier_orig, key='binary_only'):
        problems.append((X_multiclass, y_multiclass, y_names_multiclass))
    for X, y, y_names in problems:
        for y_names_i in [y_names, y_names.astype('O')]:
            y_ = _choose_check_classifiers_labels(name, y, y_names_i)
            check_classifiers_predictions(X, y_, name, classifier_orig)
    labels_binary = [-1, 1]
    y_names_binary = np.take(labels_binary, y_binary)
    y_binary = _choose_check_classifiers_labels(name, y_binary, y_names_binary)
    check_classifiers_predictions(X_binary, y_binary, name, classifier_orig)