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
def test_ovo_decision_function():
    n_samples = iris.data.shape[0]
    ovo_clf = OneVsOneClassifier(LinearSVC(dual='auto', random_state=0))
    ovo_clf.fit(iris.data, iris.target == 0)
    decisions = ovo_clf.decision_function(iris.data)
    assert decisions.shape == (n_samples,)
    ovo_clf.fit(iris.data, iris.target)
    decisions = ovo_clf.decision_function(iris.data)
    assert decisions.shape == (n_samples, n_classes)
    assert_array_equal(decisions.argmax(axis=1), ovo_clf.predict(iris.data))
    votes = np.zeros((n_samples, n_classes))
    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            pred = ovo_clf.estimators_[k].predict(iris.data)
            votes[pred == 0, i] += 1
            votes[pred == 1, j] += 1
            k += 1
    assert_array_equal(votes, np.round(decisions))
    for class_idx in range(n_classes):
        assert set(votes[:, class_idx]).issubset(set([0.0, 1.0, 2.0]))
        assert len(np.unique(decisions[:, class_idx])) > 146