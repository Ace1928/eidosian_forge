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
def test_pairwise_indices():
    clf_precomputed = svm.SVC(kernel='precomputed')
    X, y = (iris.data, iris.target)
    ovr_false = OneVsOneClassifier(clf_precomputed)
    linear_kernel = np.dot(X, X.T)
    ovr_false.fit(linear_kernel, y)
    n_estimators = len(ovr_false.estimators_)
    precomputed_indices = ovr_false.pairwise_indices_
    for idx in precomputed_indices:
        assert idx.shape[0] * n_estimators / (n_estimators - 1) == linear_kernel.shape[0]