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
@pytest.mark.parametrize('Stacker, Estimator, stack_method, final_estimator, X, y', [(StackingClassifier, DummyClassifier, 'predict_proba', LogisticRegression(random_state=42), X_iris, y_iris), (StackingRegressor, DummyRegressor, 'predict', LinearRegression(), X_diabetes, y_diabetes)])
def test_stacking_prefit(Stacker, Estimator, stack_method, final_estimator, X, y):
    """Check the behaviour of stacking when `cv='prefit'`"""
    X_train1, X_train2, y_train1, y_train2 = train_test_split(X, y, random_state=42, test_size=0.5)
    estimators = [('d0', Estimator().fit(X_train1, y_train1)), ('d1', Estimator().fit(X_train1, y_train1))]
    for _, estimator in estimators:
        estimator.fit = Mock(name='fit')
        stack_func = getattr(estimator, stack_method)
        predict_method_mocked = Mock(side_effect=stack_func)
        predict_method_mocked.__name__ = stack_method
        setattr(estimator, stack_method, predict_method_mocked)
    stacker = Stacker(estimators=estimators, cv='prefit', final_estimator=final_estimator)
    stacker.fit(X_train2, y_train2)
    assert stacker.estimators_ == [estimator for _, estimator in estimators]
    assert all((estimator.fit.call_count == 0 for estimator in stacker.estimators_))
    for estimator in stacker.estimators_:
        stack_func_mock = getattr(estimator, stack_method)
        stack_func_mock.assert_called_with(X_train2)