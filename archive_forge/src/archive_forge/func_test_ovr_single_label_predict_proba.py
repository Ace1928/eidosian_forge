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
def test_ovr_single_label_predict_proba():
    base_clf = MultinomialNB(alpha=1)
    X, Y = (iris.data, iris.target)
    X_train, Y_train = (X[:80], Y[:80])
    X_test = X[80:]
    clf = OneVsRestClassifier(base_clf).fit(X_train, Y_train)
    decision_only = OneVsRestClassifier(svm.SVR()).fit(X_train, Y_train)
    assert not hasattr(decision_only, 'predict_proba')
    Y_pred = clf.predict(X_test)
    Y_proba = clf.predict_proba(X_test)
    assert_almost_equal(Y_proba.sum(axis=1), 1.0)
    pred = Y_proba.argmax(axis=1)
    assert not (pred - Y_pred).any()