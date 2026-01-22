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
def test_learning_curve_with_shuffle():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]])
    y = np.array([1, 1, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 2, 3, 4])
    groups = np.array([1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 4, 4, 4, 4])
    estimator = PassiveAggressiveClassifier(max_iter=5, tol=None, shuffle=False)
    cv = GroupKFold(n_splits=2)
    train_sizes_batch, train_scores_batch, test_scores_batch = learning_curve(estimator, X, y, cv=cv, n_jobs=1, train_sizes=np.linspace(0.3, 1.0, 3), groups=groups, shuffle=True, random_state=2)
    assert_array_almost_equal(train_scores_batch.mean(axis=1), np.array([0.75, 0.3, 0.36111111]))
    assert_array_almost_equal(test_scores_batch.mean(axis=1), np.array([0.36111111, 0.25, 0.25]))
    with pytest.raises(ValueError):
        learning_curve(estimator, X, y, cv=cv, n_jobs=1, train_sizes=np.linspace(0.3, 1.0, 3), groups=groups, error_score='raise')
    train_sizes_inc, train_scores_inc, test_scores_inc = learning_curve(estimator, X, y, cv=cv, n_jobs=1, train_sizes=np.linspace(0.3, 1.0, 3), groups=groups, shuffle=True, random_state=2, exploit_incremental_learning=True)
    assert_array_almost_equal(train_scores_inc.mean(axis=1), train_scores_batch.mean(axis=1))
    assert_array_almost_equal(test_scores_inc.mean(axis=1), test_scores_batch.mean(axis=1))