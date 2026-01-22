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
@pytest.mark.parametrize('estimator, X, y', [(StackingClassifier(estimators=[('lr', LogisticRegression(random_state=0)), ('svm', LinearSVC(dual='auto', random_state=0))]), X_iris[:100], y_iris[:100]), (StackingRegressor(estimators=[('lr', LinearRegression()), ('svm', LinearSVR(dual='auto', random_state=0))]), X_diabetes, y_diabetes)], ids=['StackingClassifier', 'StackingRegressor'])
def test_stacking_randomness(estimator, X, y):
    estimator_full = clone(estimator)
    estimator_full.set_params(cv=KFold(shuffle=True, random_state=np.random.RandomState(0)))
    estimator_drop = clone(estimator)
    estimator_drop.set_params(lr='drop')
    estimator_drop.set_params(cv=KFold(shuffle=True, random_state=np.random.RandomState(0)))
    assert_allclose(estimator_full.fit(X, y).transform(X)[:, 1:], estimator_drop.fit(X, y).transform(X))