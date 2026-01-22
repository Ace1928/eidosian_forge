import importlib
import sys
import unittest
import warnings
from numbers import Integral, Real
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn import config_context, get_config
from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_multilabel_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning, SkipTestWarning
from sklearn.linear_model import (
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC, NuSVC
from sklearn.utils import _array_api, all_estimators, deprecated
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
def test_check_classifiers_multilabel_output_format_predict():
    n_samples, test_size, n_outputs = (100, 25, 5)
    _, y = make_multilabel_classification(n_samples=n_samples, n_features=2, n_classes=n_outputs, n_labels=3, length=50, allow_unlabeled=True, random_state=0)
    y_test = y[-test_size:]

    class MultiLabelClassifierPredict(_BaseMultiLabelClassifierMock):

        def predict(self, X):
            return self.response_output
    clf = MultiLabelClassifierPredict(response_output=y_test.tolist())
    err_msg = "MultiLabelClassifierPredict.predict is expected to output a NumPy array. Got <class 'list'> instead."
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict(clf.__class__.__name__, clf)
    clf = MultiLabelClassifierPredict(response_output=y_test[:, :-1])
    err_msg = 'MultiLabelClassifierPredict.predict outputs a NumPy array of shape \\(25, 4\\) instead of \\(25, 5\\).'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict(clf.__class__.__name__, clf)
    clf = MultiLabelClassifierPredict(response_output=y_test.astype(np.float64))
    err_msg = 'MultiLabelClassifierPredict.predict does not output the same dtype than the targets.'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict(clf.__class__.__name__, clf)