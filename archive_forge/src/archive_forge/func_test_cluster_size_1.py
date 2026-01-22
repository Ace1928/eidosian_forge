import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._unsupervised import _silhouette_reduce
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import (
def test_cluster_size_1():
    X = [[0.0], [1.0], [1.0], [2.0], [3.0], [3.0]]
    labels = np.array([0, 1, 1, 1, 2, 2])
    silhouette = silhouette_score(X, labels)
    assert not np.isnan(silhouette)
    ss = silhouette_samples(X, labels)
    assert_array_equal(ss, [0, 0.5, 0.5, 0, 1, 1])