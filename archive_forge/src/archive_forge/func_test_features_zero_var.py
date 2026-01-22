import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.neighbors import NearestCentroid
from sklearn.utils.fixes import CSR_CONTAINERS
def test_features_zero_var():
    X = np.empty((10, 2))
    X[:, 0] = -0.13725701
    X[:, 1] = -0.9853293
    y = np.zeros(10)
    y[0] = 1
    clf = NearestCentroid(shrink_threshold=0.1)
    with pytest.raises(ValueError):
        clf.fit(X, y)