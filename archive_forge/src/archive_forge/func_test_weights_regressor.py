import re
import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_multilabel_classification
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import (
def test_weights_regressor():
    """Check weighted average regression prediction on diabetes dataset."""
    reg1 = DummyRegressor(strategy='mean')
    reg2 = DummyRegressor(strategy='median')
    reg3 = DummyRegressor(strategy='quantile', quantile=0.2)
    ereg = VotingRegressor([('mean', reg1), ('median', reg2), ('quantile', reg3)], weights=[1, 2, 10])
    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_r, y_r, test_size=0.25)
    reg1_pred = reg1.fit(X_r_train, y_r_train).predict(X_r_test)
    reg2_pred = reg2.fit(X_r_train, y_r_train).predict(X_r_test)
    reg3_pred = reg3.fit(X_r_train, y_r_train).predict(X_r_test)
    ereg_pred = ereg.fit(X_r_train, y_r_train).predict(X_r_test)
    avg = np.average(np.asarray([reg1_pred, reg2_pred, reg3_pred]), axis=0, weights=[1, 2, 10])
    assert_almost_equal(ereg_pred, avg, decimal=2)
    ereg_weights_none = VotingRegressor([('mean', reg1), ('median', reg2), ('quantile', reg3)], weights=None)
    ereg_weights_equal = VotingRegressor([('mean', reg1), ('median', reg2), ('quantile', reg3)], weights=[1, 1, 1])
    ereg_weights_none.fit(X_r_train, y_r_train)
    ereg_weights_equal.fit(X_r_train, y_r_train)
    ereg_none_pred = ereg_weights_none.predict(X_r_test)
    ereg_equal_pred = ereg_weights_equal.predict(X_r_test)
    assert_almost_equal(ereg_none_pred, ereg_equal_pred, decimal=2)