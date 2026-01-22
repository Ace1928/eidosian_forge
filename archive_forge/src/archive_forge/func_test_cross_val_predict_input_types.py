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
def test_cross_val_predict_input_types(coo_container):
    iris = load_iris()
    X, y = (iris.data, iris.target)
    X_sparse = coo_container(X)
    multioutput_y = np.column_stack([y, y[::-1]])
    clf = Ridge(fit_intercept=False, random_state=0)
    predictions = cross_val_predict(clf, X, y)
    assert predictions.shape == (150,)
    predictions = cross_val_predict(clf, X_sparse, multioutput_y)
    assert predictions.shape == (150, 2)
    predictions = cross_val_predict(clf, X_sparse, y)
    assert_array_equal(predictions.shape, (150,))
    predictions = cross_val_predict(clf, X_sparse, multioutput_y)
    assert_array_equal(predictions.shape, (150, 2))
    list_check = lambda x: isinstance(x, list)
    clf = CheckingClassifier(check_X=list_check)
    predictions = cross_val_predict(clf, X.tolist(), y.tolist())
    clf = CheckingClassifier(check_y=list_check)
    predictions = cross_val_predict(clf, X, y.tolist())
    predictions = cross_val_predict(LogisticRegression(solver='liblinear'), X.tolist(), y.tolist(), method='decision_function')
    predictions = cross_val_predict(LogisticRegression(solver='liblinear'), X, y.tolist(), method='decision_function')
    X_3d = X[:, :, np.newaxis]
    check_3d = lambda x: x.ndim == 3
    clf = CheckingClassifier(check_X=check_3d)
    predictions = cross_val_predict(clf, X_3d, y)
    assert_array_equal(predictions.shape, (150,))