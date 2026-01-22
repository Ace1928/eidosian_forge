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
def test_ovo_partial_fit_predict():
    temp = datasets.load_iris()
    X, y = (temp.data, temp.target)
    ovo1 = OneVsOneClassifier(MultinomialNB())
    ovo1.partial_fit(X[:100], y[:100], np.unique(y))
    ovo1.partial_fit(X[100:], y[100:])
    pred1 = ovo1.predict(X)
    ovo2 = OneVsOneClassifier(MultinomialNB())
    ovo2.fit(X, y)
    pred2 = ovo2.predict(X)
    assert len(ovo1.estimators_) == n_classes * (n_classes - 1) / 2
    assert np.mean(y == pred1) > 0.65
    assert_almost_equal(pred1, pred2)
    ovo1 = OneVsOneClassifier(MultinomialNB())
    ovo1.partial_fit(X[:60], y[:60], np.unique(y))
    ovo1.partial_fit(X[60:], y[60:])
    pred1 = ovo1.predict(X)
    ovo2 = OneVsOneClassifier(MultinomialNB())
    pred2 = ovo2.fit(X, y).predict(X)
    assert_almost_equal(pred1, pred2)
    assert len(ovo1.estimators_) == len(np.unique(y))
    assert np.mean(y == pred1) > 0.65
    ovo = OneVsOneClassifier(MultinomialNB())
    X = np.random.rand(14, 2)
    y = [1, 1, 2, 3, 3, 0, 0, 4, 4, 4, 4, 4, 2, 2]
    ovo.partial_fit(X[:7], y[:7], [0, 1, 2, 3, 4])
    ovo.partial_fit(X[7:], y[7:])
    pred = ovo.predict(X)
    ovo2 = OneVsOneClassifier(MultinomialNB())
    pred2 = ovo2.fit(X, y).predict(X)
    assert_almost_equal(pred, pred2)
    ovo = OneVsOneClassifier(MultinomialNB())
    error_y = [0, 1, 2, 3, 4, 5, 2]
    message_re = escape('Mini-batch contains {0} while it must be subset of {1}'.format(np.unique(error_y), np.unique(y)))
    with pytest.raises(ValueError, match=message_re):
        ovo.partial_fit(X[:7], error_y, np.unique(y))
    ovr = OneVsOneClassifier(SVC())
    assert not hasattr(ovr, 'partial_fit')