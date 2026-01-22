import re
import warnings
from itertools import product
import joblib
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import (
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning, NotFittedError
from sklearn.metrics._dist_metrics import (
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, pairwise_distances
from sklearn.metrics.tests.test_dist_metrics import BOOL_METRICS
from sklearn.metrics.tests.test_pairwise_distances_reduction import (
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import (
from sklearn.neighbors._base import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.validation import check_random_state
def test_neighbors_digits():
    X = digits.data.astype('uint8')
    Y = digits.target
    n_samples, n_features = X.shape
    train_test_boundary = int(n_samples * 0.8)
    train = np.arange(0, train_test_boundary)
    test = np.arange(train_test_boundary, n_samples)
    X_train, Y_train, X_test, Y_test = (X[train], Y[train], X[test], Y[test])
    clf = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='brute')
    score_uint8 = clf.fit(X_train, Y_train).score(X_test, Y_test)
    score_float = clf.fit(X_train.astype(float, copy=False), Y_train).score(X_test.astype(float, copy=False), Y_test)
    assert score_uint8 == score_float