from unittest.mock import Mock
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.datasets import (
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.linear_model import (
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.svm import SVC, LinearSVC, LinearSVR
from sklearn.utils._mocking import CheckingClassifier
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('stack_method', ['auto', 'predict'])
@pytest.mark.parametrize('passthrough', [False, True])
def test_stacking_classifier_multilabel_auto_predict(stack_method, passthrough):
    """Check the behaviour for the multilabel classification case for stack methods
    supported for all estimators or automatically picked up.
    """
    X_train, X_test, y_train, y_test = train_test_split(X_multilabel, y_multilabel, stratify=y_multilabel, random_state=42)
    y_train_before_fit = y_train.copy()
    n_outputs = 3
    estimators = [('mlp', MLPClassifier(random_state=42)), ('rf', RandomForestClassifier(random_state=42)), ('ridge', RidgeClassifier())]
    final_estimator = KNeighborsClassifier()
    clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator, passthrough=passthrough, stack_method=stack_method).fit(X_train, y_train)
    assert_array_equal(y_train_before_fit, y_train)
    y_pred = clf.predict(X_test)
    assert y_pred.shape == y_test.shape
    if stack_method == 'auto':
        expected_stack_methods = ['predict_proba', 'predict_proba', 'decision_function']
    else:
        expected_stack_methods = ['predict'] * len(estimators)
    assert clf.stack_method_ == expected_stack_methods
    n_features_X_trans = n_outputs * len(estimators)
    if passthrough:
        n_features_X_trans += X_train.shape[1]
    X_trans = clf.transform(X_test)
    assert X_trans.shape == (X_test.shape[0], n_features_X_trans)
    assert_array_equal(clf.classes_, [np.array([0, 1])] * n_outputs)