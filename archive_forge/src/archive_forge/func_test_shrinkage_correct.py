import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.neighbors import NearestCentroid
from sklearn.utils.fixes import CSR_CONTAINERS
def test_shrinkage_correct():
    X = np.array([[0, 1], [1, 0], [1, 1], [2, 0], [6, 8]])
    y = np.array([1, 1, 2, 2, 2])
    clf = NearestCentroid(shrink_threshold=0.1)
    clf.fit(X, y)
    expected_result = np.array([[0.778731, 0.8545292], [2.814179, 2.763647]])
    np.testing.assert_array_almost_equal(clf.centroids_, expected_result)