import re
import numpy as np
import pytest
from numpy.testing import (
from sklearn import base, datasets, linear_model, metrics, svm
from sklearn.datasets import make_blobs, make_classification
from sklearn.exceptions import (
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import (  # type: ignore
from sklearn.svm._classes import _validate_dual_parameter
from sklearn.utils import check_random_state, shuffle
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.validation import _num_samples
@pytest.mark.parametrize('SVM, params', [(LinearSVC, {'penalty': 'l1', 'loss': 'squared_hinge', 'dual': False}), (LinearSVC, {'penalty': 'l2', 'loss': 'squared_hinge', 'dual': True}), (LinearSVC, {'penalty': 'l2', 'loss': 'squared_hinge', 'dual': False}), (LinearSVC, {'penalty': 'l2', 'loss': 'hinge', 'dual': True}), (LinearSVR, {'loss': 'epsilon_insensitive', 'dual': True}), (LinearSVR, {'loss': 'squared_epsilon_insensitive', 'dual': True}), (LinearSVR, {'loss': 'squared_epsilon_insensitive', 'dual': True})])
def test_linearsvm_liblinear_sample_weight(SVM, params):
    X = np.array([[1, 3], [1, 3], [1, 3], [1, 3], [2, 1], [2, 1], [2, 1], [2, 1], [3, 3], [3, 3], [3, 3], [3, 3], [4, 1], [4, 1], [4, 1], [4, 1]], dtype=np.dtype('float'))
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.dtype('int'))
    X2 = np.vstack([X, X])
    y2 = np.hstack([y, 3 - y])
    sample_weight = np.ones(shape=len(y) * 2)
    sample_weight[len(y):] = 0
    X2, y2, sample_weight = shuffle(X2, y2, sample_weight, random_state=0)
    base_estimator = SVM(random_state=42)
    base_estimator.set_params(**params)
    base_estimator.set_params(tol=1e-12, max_iter=1000)
    est_no_weight = base.clone(base_estimator).fit(X, y)
    est_with_weight = base.clone(base_estimator).fit(X2, y2, sample_weight=sample_weight)
    for method in ('predict', 'decision_function'):
        if hasattr(base_estimator, method):
            X_est_no_weight = getattr(est_no_weight, method)(X)
            X_est_with_weight = getattr(est_with_weight, method)(X)
            assert_allclose(X_est_no_weight, X_est_with_weight)