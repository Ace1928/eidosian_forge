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
@pytest.mark.parametrize('sparse_container', COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS)
def test_stacking_regressor_sparse_passthrough(sparse_container):
    X_train, X_test, y_train, _ = train_test_split(sparse_container(scale(X_diabetes)), y_diabetes, random_state=42)
    estimators = [('lr', LinearRegression()), ('svr', LinearSVR(dual='auto'))]
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    clf = StackingRegressor(estimators=estimators, final_estimator=rf, cv=5, passthrough=True)
    clf.fit(X_train, y_train)
    X_trans = clf.transform(X_test)
    assert_allclose_dense_sparse(X_test, X_trans[:, -10:])
    assert sparse.issparse(X_trans)
    assert X_test.format == X_trans.format