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
def check_classifiers_one_label_sample_weights(name, classifier_orig):
    """Check that classifiers accepting sample_weight fit or throws a ValueError with
    an explicit message if the problem is reduced to one class.
    """
    error_fit = f"{name} failed when fitted on one label after sample_weight trimming. Error message is not explicit, it should have 'class'."
    error_predict = f'{name} prediction results should only output the remaining class.'
    rnd = np.random.RandomState(0)
    X_train = rnd.uniform(size=(10, 10))
    X_test = rnd.uniform(size=(10, 10))
    y = np.arange(10) % 2
    sample_weight = y.copy()
    classifier = clone(classifier_orig)
    if has_fit_parameter(classifier, 'sample_weight'):
        match = ['\\bclass(es)?\\b', error_predict]
        err_type, err_msg = ((AssertionError, ValueError), error_fit)
    else:
        match = '\\bsample_weight\\b'
        err_type, err_msg = ((TypeError, ValueError), None)
    with raises(err_type, match=match, may_pass=True, err_msg=err_msg) as cm:
        classifier.fit(X_train, y, sample_weight=sample_weight)
        if cm.raised_and_matched:
            return
        assert_array_equal(classifier.predict(X_test), np.ones(10), err_msg=error_predict)