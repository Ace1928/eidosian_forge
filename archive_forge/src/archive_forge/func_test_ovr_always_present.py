from re import escape
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from sklearn import datasets, svm
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.multiclass import (
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import (
from sklearn.utils._mocking import CheckingClassifier
from sklearn.utils._testing import assert_almost_equal, assert_array_equal
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import check_classification_targets, type_of_target
def test_ovr_always_present():
    X = np.ones((10, 2))
    X[:5, :] = 0
    y = np.zeros((10, 3))
    y[5:, 0] = 1
    y[:, 1] = 1
    y[:, 2] = 1
    ovr = OneVsRestClassifier(LogisticRegression())
    msg = 'Label .+ is present in all training examples'
    with pytest.warns(UserWarning, match=msg):
        ovr.fit(X, y)
    y_pred = ovr.predict(X)
    assert_array_equal(np.array(y_pred), np.array(y))
    y_pred = ovr.decision_function(X)
    assert np.unique(y_pred[:, -2:]) == 1
    y_pred = ovr.predict_proba(X)
    assert_array_equal(y_pred[:, -1], np.ones(X.shape[0]))
    y = np.zeros((10, 2))
    y[5:, 0] = 1
    ovr = OneVsRestClassifier(LogisticRegression())
    msg = 'Label not 1 is present in all training examples'
    with pytest.warns(UserWarning, match=msg):
        ovr.fit(X, y)
    y_pred = ovr.predict_proba(X)
    assert_array_equal(y_pred[:, -1], np.zeros(X.shape[0]))