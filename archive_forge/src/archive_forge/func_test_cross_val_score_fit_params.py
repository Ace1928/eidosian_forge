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
def test_cross_val_score_fit_params(coo_container):
    clf = MockClassifier()
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    W_sparse = coo_container((np.array([1]), (np.array([1]), np.array([0]))), shape=(10, 1))
    P_sparse = coo_container(np.eye(5))
    DUMMY_INT = 42
    DUMMY_STR = '42'
    DUMMY_OBJ = object()

    def assert_fit_params(clf):
        assert clf.dummy_int == DUMMY_INT
        assert clf.dummy_str == DUMMY_STR
        assert clf.dummy_obj == DUMMY_OBJ
    fit_params = {'sample_weight': np.ones(n_samples), 'class_prior': np.full(n_classes, 1.0 / n_classes), 'sparse_sample_weight': W_sparse, 'sparse_param': P_sparse, 'dummy_int': DUMMY_INT, 'dummy_str': DUMMY_STR, 'dummy_obj': DUMMY_OBJ, 'callback': assert_fit_params}
    cross_val_score(clf, X, y, params=fit_params)