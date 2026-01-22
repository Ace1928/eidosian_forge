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
def test_radius_neighbors_classifier_zero_distance():
    X = np.array([[1.0, 1.0], [2.0, 2.0]])
    y = np.array([1, 2])
    radius = 0.1
    z1 = np.array([[1.01, 1.01], [2.0, 2.0]])
    correct_labels1 = np.array([1, 2])
    weight_func = _weight_func
    for algorithm in ALGORITHMS:
        for weights in ['uniform', 'distance', weight_func]:
            clf = neighbors.RadiusNeighborsClassifier(radius=radius, weights=weights, algorithm=algorithm)
            clf.fit(X, y)
            with np.errstate(invalid='ignore'):
                assert_array_equal(correct_labels1, clf.predict(z1))