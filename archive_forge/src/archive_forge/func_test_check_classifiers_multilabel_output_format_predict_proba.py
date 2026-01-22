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
def test_check_classifiers_multilabel_output_format_predict_proba():
    n_samples, test_size, n_outputs = (100, 25, 5)
    _, y = make_multilabel_classification(n_samples=n_samples, n_features=2, n_classes=n_outputs, n_labels=3, length=50, allow_unlabeled=True, random_state=0)
    y_test = y[-test_size:]

    class MultiLabelClassifierPredictProba(_BaseMultiLabelClassifierMock):

        def predict_proba(self, X):
            return self.response_output
    for csr_container in CSR_CONTAINERS:
        clf = MultiLabelClassifierPredictProba(response_output=csr_container(y_test))
        err_msg = f'Unknown returned type .*{csr_container.__name__}.* by MultiLabelClassifierPredictProba.predict_proba. A list or a Numpy array is expected.'
        with raises(ValueError, match=err_msg):
            check_classifiers_multilabel_output_format_predict_proba(clf.__class__.__name__, clf)
    clf = MultiLabelClassifierPredictProba(response_output=y_test.tolist())
    err_msg = f'When MultiLabelClassifierPredictProba.predict_proba returns a list, the list should be of length n_outputs and contain NumPy arrays. Got length of {test_size} instead of {n_outputs}.'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(clf.__class__.__name__, clf)
    response_output = [np.ones_like(y_test) for _ in range(n_outputs)]
    clf = MultiLabelClassifierPredictProba(response_output=response_output)
    err_msg = 'When MultiLabelClassifierPredictProba.predict_proba returns a list, this list should contain NumPy arrays of shape \\(n_samples, 2\\). Got NumPy arrays of shape \\(25, 5\\) instead of \\(25, 2\\).'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(clf.__class__.__name__, clf)
    response_output = [np.ones(shape=(y_test.shape[0], 2), dtype=np.int64) for _ in range(n_outputs)]
    clf = MultiLabelClassifierPredictProba(response_output=response_output)
    err_msg = 'When MultiLabelClassifierPredictProba.predict_proba returns a list, it should contain NumPy arrays with floating dtype.'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(clf.__class__.__name__, clf)
    response_output = [np.ones(shape=(y_test.shape[0], 2), dtype=np.float64) for _ in range(n_outputs)]
    clf = MultiLabelClassifierPredictProba(response_output=response_output)
    err_msg = 'When MultiLabelClassifierPredictProba.predict_proba returns a list, each NumPy array should contain probabilities for each class and thus each row should sum to 1'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(clf.__class__.__name__, clf)
    clf = MultiLabelClassifierPredictProba(response_output=y_test[:, :-1])
    err_msg = 'When MultiLabelClassifierPredictProba.predict_proba returns a NumPy array, the expected shape is \\(n_samples, n_outputs\\). Got \\(25, 4\\) instead of \\(25, 5\\).'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(clf.__class__.__name__, clf)
    response_output = np.zeros_like(y_test, dtype=np.int64)
    clf = MultiLabelClassifierPredictProba(response_output=response_output)
    err_msg = 'When MultiLabelClassifierPredictProba.predict_proba returns a NumPy array, the expected data type is floating.'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(clf.__class__.__name__, clf)
    clf = MultiLabelClassifierPredictProba(response_output=y_test * 2.0)
    err_msg = 'When MultiLabelClassifierPredictProba.predict_proba returns a NumPy array, this array is expected to provide probabilities of the positive class and should therefore contain values between 0 and 1.'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(clf.__class__.__name__, clf)