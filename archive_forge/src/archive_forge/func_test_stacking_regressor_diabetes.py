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
@pytest.mark.parametrize('cv', [3, KFold(n_splits=3, shuffle=True, random_state=42)])
@pytest.mark.parametrize('final_estimator, predict_params', [(None, {}), (RandomForestRegressor(random_state=42), {}), (DummyRegressor(), {'return_std': True})])
@pytest.mark.parametrize('passthrough', [False, True])
def test_stacking_regressor_diabetes(cv, final_estimator, predict_params, passthrough):
    X_train, X_test, y_train, _ = train_test_split(scale(X_diabetes), y_diabetes, random_state=42)
    estimators = [('lr', LinearRegression()), ('svr', LinearSVR(dual='auto'))]
    reg = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=cv, passthrough=passthrough)
    reg.fit(X_train, y_train)
    result = reg.predict(X_test, **predict_params)
    expected_result_length = 2 if predict_params else 1
    if predict_params:
        assert len(result) == expected_result_length
    X_trans = reg.transform(X_test)
    expected_column_count = 12 if passthrough else 2
    assert X_trans.shape[1] == expected_column_count
    if passthrough:
        assert_allclose(X_test, X_trans[:, -10:])
    reg.set_params(lr='drop')
    reg.fit(X_train, y_train)
    reg.predict(X_test)
    X_trans = reg.transform(X_test)
    expected_column_count_drop = 11 if passthrough else 1
    assert X_trans.shape[1] == expected_column_count_drop
    if passthrough:
        assert_allclose(X_test, X_trans[:, -10:])