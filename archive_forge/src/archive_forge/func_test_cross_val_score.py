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
def test_cross_val_score(coo_container):
    clf = MockClassifier()
    X_sparse = coo_container(X)
    for a in range(-10, 10):
        clf.a = a
        scores = cross_val_score(clf, X, y2)
        assert_array_equal(scores, clf.score(X, y2))
        multioutput_y = np.column_stack([y2, y2[::-1]])
        scores = cross_val_score(clf, X_sparse, multioutput_y)
        assert_array_equal(scores, clf.score(X_sparse, multioutput_y))
        scores = cross_val_score(clf, X_sparse, y2)
        assert_array_equal(scores, clf.score(X_sparse, y2))
        scores = cross_val_score(clf, X_sparse, multioutput_y)
        assert_array_equal(scores, clf.score(X_sparse, multioutput_y))
    list_check = lambda x: isinstance(x, list)
    clf = CheckingClassifier(check_X=list_check)
    scores = cross_val_score(clf, X.tolist(), y2.tolist(), cv=3)
    clf = CheckingClassifier(check_y=list_check)
    scores = cross_val_score(clf, X, y2.tolist(), cv=3)
    X_3d = X[:, :, np.newaxis]
    clf = MockClassifier(allow_nd=True)
    scores = cross_val_score(clf, X_3d, y2)
    clf = MockClassifier(allow_nd=False)
    with pytest.raises(ValueError):
        cross_val_score(clf, X_3d, y2, error_score='raise')