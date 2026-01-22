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
@pytest.mark.parametrize('stacker, X, y', [(StackingClassifier(estimators=[('lr', LogisticRegression()), ('svm', LinearSVC(dual='auto', random_state=42))], final_estimator=LogisticRegression(), cv=KFold(shuffle=True, random_state=42)), *load_breast_cancer(return_X_y=True)), (StackingRegressor(estimators=[('lr', LinearRegression()), ('svm', LinearSVR(dual='auto', random_state=42))], final_estimator=LinearRegression(), cv=KFold(shuffle=True, random_state=42)), X_diabetes, y_diabetes)], ids=['StackingClassifier', 'StackingRegressor'])
def test_stacking_with_sample_weight(stacker, X, y):
    n_half_samples = len(y) // 2
    total_sample_weight = np.array([0.1] * n_half_samples + [0.9] * (len(y) - n_half_samples))
    X_train, X_test, y_train, _, sample_weight_train, _ = train_test_split(X, y, total_sample_weight, random_state=42)
    with ignore_warnings(category=ConvergenceWarning):
        stacker.fit(X_train, y_train)
    y_pred_no_weight = stacker.predict(X_test)
    with ignore_warnings(category=ConvergenceWarning):
        stacker.fit(X_train, y_train, sample_weight=np.ones(y_train.shape))
    y_pred_unit_weight = stacker.predict(X_test)
    assert_allclose(y_pred_no_weight, y_pred_unit_weight)
    with ignore_warnings(category=ConvergenceWarning):
        stacker.fit(X_train, y_train, sample_weight=sample_weight_train)
    y_pred_biased = stacker.predict(X_test)
    assert np.abs(y_pred_no_weight - y_pred_biased).sum() > 0