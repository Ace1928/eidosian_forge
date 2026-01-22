import os
import re
import sys
import tempfile
import warnings
from functools import partial
from io import StringIO
from time import sleep
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import (
from sklearn.model_selection import (
from sklearn.model_selection._validation import (
from sklearn.model_selection.tests.common import OneTimeSplitter
from sklearn.model_selection.tests.test_search import FailingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.svm import SVC, LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.utils import shuffle
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
@pytest.mark.parametrize('coo_container', COO_CONTAINERS)
def test_permutation_score(coo_container):
    iris = load_iris()
    X = iris.data
    X_sparse = coo_container(X)
    y = iris.target
    svm = SVC(kernel='linear')
    cv = StratifiedKFold(2)
    score, scores, pvalue = permutation_test_score(svm, X, y, n_permutations=30, cv=cv, scoring='accuracy')
    assert score > 0.9
    assert_almost_equal(pvalue, 0.0, 1)
    score_group, _, pvalue_group = permutation_test_score(svm, X, y, n_permutations=30, cv=cv, scoring='accuracy', groups=np.ones(y.size), random_state=0)
    assert score_group == score
    assert pvalue_group == pvalue
    svm_sparse = SVC(kernel='linear')
    cv_sparse = StratifiedKFold(2)
    score_group, _, pvalue_group = permutation_test_score(svm_sparse, X_sparse, y, n_permutations=30, cv=cv_sparse, scoring='accuracy', groups=np.ones(y.size), random_state=0)
    assert score_group == score
    assert pvalue_group == pvalue

    def custom_score(y_true, y_pred):
        return ((y_true == y_pred).sum() - (y_true != y_pred).sum()) / y_true.shape[0]
    scorer = make_scorer(custom_score)
    score, _, pvalue = permutation_test_score(svm, X, y, n_permutations=100, scoring=scorer, cv=cv, random_state=0)
    assert_almost_equal(score, 0.93, 2)
    assert_almost_equal(pvalue, 0.01, 3)
    y = np.mod(np.arange(len(y)), 3)
    score, scores, pvalue = permutation_test_score(svm, X, y, n_permutations=30, cv=cv, scoring='accuracy')
    assert score < 0.5
    assert pvalue > 0.2