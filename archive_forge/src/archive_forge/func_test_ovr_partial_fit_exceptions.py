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
def test_ovr_partial_fit_exceptions():
    ovr = OneVsRestClassifier(MultinomialNB())
    X = np.abs(np.random.randn(14, 2))
    y = [1, 1, 1, 1, 2, 3, 3, 0, 0, 2, 3, 1, 2, 3]
    ovr.partial_fit(X[:7], y[:7], np.unique(y))
    y1 = [5] + y[7:-1]
    msg = 'Mini-batch contains \\[.+\\] while classes must be subset of \\[.+\\]'
    with pytest.raises(ValueError, match=msg):
        ovr.partial_fit(X=X[7:], y=y1)