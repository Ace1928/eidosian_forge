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
def test_radius_neighbors_classifier_outlier_labeling(global_dtype, algorithm, weights):
    X = np.array([[1.0, 1.0], [2.0, 2.0], [0.99, 0.99], [0.98, 0.98], [2.01, 2.01]], dtype=global_dtype)
    y = np.array([1, 2, 1, 1, 2])
    radius = 0.1
    z1 = np.array([[1.01, 1.01], [2.01, 2.01]], dtype=global_dtype)
    z2 = np.array([[1.4, 1.4], [1.01, 1.01], [2.01, 2.01]], dtype=global_dtype)
    correct_labels1 = np.array([1, 2])
    correct_labels2 = np.array([-1, 1, 2])
    outlier_proba = np.array([0, 0])
    clf = neighbors.RadiusNeighborsClassifier(radius=radius, weights=weights, algorithm=algorithm, outlier_label=-1)
    clf.fit(X, y)
    assert_array_equal(correct_labels1, clf.predict(z1))
    with pytest.warns(UserWarning, match='Outlier label -1 is not in training classes'):
        assert_array_equal(correct_labels2, clf.predict(z2))
    with pytest.warns(UserWarning, match='Outlier label -1 is not in training classes'):
        assert_allclose(outlier_proba, clf.predict_proba(z2)[0])
    RNC = neighbors.RadiusNeighborsClassifier
    X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=global_dtype)
    y = np.array([0, 2, 2, 1, 1, 1, 3, 3, 3, 3])

    def check_array_exception():
        clf = RNC(radius=1, outlier_label=[[5]])
        clf.fit(X, y)
    with pytest.raises(TypeError):
        check_array_exception()

    def check_dtype_exception():
        clf = RNC(radius=1, outlier_label='a')
        clf.fit(X, y)
    with pytest.raises(TypeError):
        check_dtype_exception()
    clf = RNC(radius=1, outlier_label='most_frequent')
    clf.fit(X, y)
    proba = clf.predict_proba([[1], [15]])
    assert_array_equal(proba[1, :], [0, 0, 0, 1])
    clf = RNC(radius=1, outlier_label=1)
    clf.fit(X, y)
    proba = clf.predict_proba([[1], [15]])
    assert_array_equal(proba[1, :], [0, 1, 0, 0])
    pred = clf.predict([[1], [15]])
    assert_array_equal(pred, [2, 1])

    def check_warning():
        clf = RNC(radius=1, outlier_label=4)
        clf.fit(X, y)
        clf.predict_proba([[1], [15]])
    with pytest.warns(UserWarning):
        check_warning()
    y_multi = [[0, 1], [2, 1], [2, 2], [1, 2], [1, 2], [1, 3], [3, 3], [3, 3], [3, 0], [3, 0]]
    clf = RNC(radius=1, outlier_label=1)
    clf.fit(X, y_multi)
    proba = clf.predict_proba([[7], [15]])
    assert_array_equal(proba[1][1, :], [0, 1, 0, 0])
    pred = clf.predict([[7], [15]])
    assert_array_equal(pred[1, :], [1, 1])
    y_multi = [[0, 0], [2, 2], [2, 2], [1, 1], [1, 1], [1, 1], [3, 3], [3, 3], [3, 3], [3, 3]]
    clf = RNC(radius=1, outlier_label=[0, 1])
    clf.fit(X, y_multi)
    proba = clf.predict_proba([[7], [15]])
    assert_array_equal(proba[0][1, :], [1, 0, 0, 0])
    assert_array_equal(proba[1][1, :], [0, 1, 0, 0])
    pred = clf.predict([[7], [15]])
    assert_array_equal(pred[1, :], [0, 1])

    def check_exception():
        clf = RNC(radius=1, outlier_label=[0, 1, 2])
        clf.fit(X, y_multi)
    with pytest.raises(ValueError):
        check_exception()