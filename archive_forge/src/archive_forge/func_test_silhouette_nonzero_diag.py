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
@pytest.mark.parametrize('dtype', (np.float32, np.float64))
def test_silhouette_nonzero_diag(dtype):
    dists = pairwise_distances(np.array([[0.2, 0.1, 0.12, 1.34, 1.11, 1.6]], dtype=dtype).T)
    labels = [0, 0, 0, 1, 1, 1]
    dists[2][2] = np.finfo(dists.dtype).eps * 10
    silhouette_samples(dists, labels, metric='precomputed')
    dists[2][2] = np.finfo(dists.dtype).eps * 1000
    with pytest.raises(ValueError, match='contains non-zero'):
        silhouette_samples(dists, labels, metric='precomputed')