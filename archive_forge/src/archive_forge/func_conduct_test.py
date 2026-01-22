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
def conduct_test(base_clf, test_predict_proba=False):
    clf = OneVsRestClassifier(base_clf).fit(X, y)
    assert set(clf.classes_) == classes
    y_pred = clf.predict(np.array([[0, 0, 4]]))[0]
    assert_array_equal(y_pred, ['eggs'])
    if hasattr(base_clf, 'decision_function'):
        dec = clf.decision_function(X)
        assert dec.shape == (5,)
    if test_predict_proba:
        X_test = np.array([[0, 0, 4]])
        probabilities = clf.predict_proba(X_test)
        assert 2 == len(probabilities[0])
        assert clf.classes_[np.argmax(probabilities, axis=1)] == clf.predict(X_test)
    clf = OneVsRestClassifier(base_clf).fit(X, Y)
    y_pred = clf.predict([[3, 0, 0]])[0]
    assert y_pred == 1