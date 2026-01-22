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
def test_ovr_partial_fit():
    X, y = shuffle(iris.data, iris.target, random_state=0)
    ovr = OneVsRestClassifier(MultinomialNB())
    ovr.partial_fit(X[:100], y[:100], np.unique(y))
    ovr.partial_fit(X[100:], y[100:])
    pred = ovr.predict(X)
    ovr2 = OneVsRestClassifier(MultinomialNB())
    pred2 = ovr2.fit(X, y).predict(X)
    assert_almost_equal(pred, pred2)
    assert len(ovr.estimators_) == len(np.unique(y))
    assert np.mean(y == pred) > 0.65
    X = np.abs(np.random.randn(14, 2))
    y = [1, 1, 1, 1, 2, 3, 3, 0, 0, 2, 3, 1, 2, 3]
    ovr = OneVsRestClassifier(SGDClassifier(max_iter=1, tol=None, shuffle=False, random_state=0))
    ovr.partial_fit(X[:7], y[:7], np.unique(y))
    ovr.partial_fit(X[7:], y[7:])
    pred = ovr.predict(X)
    ovr1 = OneVsRestClassifier(SGDClassifier(max_iter=1, tol=None, shuffle=False, random_state=0))
    pred1 = ovr1.fit(X, y).predict(X)
    assert np.mean(pred == y) == np.mean(pred1 == y)
    ovr = OneVsRestClassifier(SVC())
    assert not hasattr(ovr, 'partial_fit')