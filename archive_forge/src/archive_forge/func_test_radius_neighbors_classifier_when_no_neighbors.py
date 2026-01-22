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
@pytest.mark.parametrize('algorithm', ALGORITHMS)
@pytest.mark.parametrize('weights', WEIGHTS)
@pytest.mark.parametrize('outlier_label', [0, -1, None])
def test_radius_neighbors_classifier_when_no_neighbors(global_dtype, algorithm, weights, outlier_label):
    X = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=global_dtype)
    y = np.array([1, 2])
    radius = 0.1
    z1 = np.array([[1.01, 1.01], [2.01, 2.01]], dtype=global_dtype)
    z2 = np.array([[1.01, 1.01], [1.4, 1.4]], dtype=global_dtype)
    rnc = neighbors.RadiusNeighborsClassifier
    clf = rnc(radius=radius, weights=weights, algorithm=algorithm, outlier_label=outlier_label)
    clf.fit(X, y)
    assert_array_equal(np.array([1, 2]), clf.predict(z1))
    if outlier_label is None:
        with pytest.raises(ValueError):
            clf.predict(z2)