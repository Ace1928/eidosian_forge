import pickle
import re
import numpy as np
import pytest
from scipy.linalg import LinAlgError
from sklearn.cluster import SpectralClustering, spectral_clustering
from sklearn.cluster._spectral import cluster_qr, discretize
from sklearn.datasets import make_blobs
from sklearn.feature_extraction import img_to_graph
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import kernel_metrics, rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
def test_affinities():
    X, y = make_blobs(n_samples=20, random_state=0, centers=[[1, 1], [-1, -1]], cluster_std=0.01)
    sp = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0)
    with pytest.warns(UserWarning, match='not fully connected'):
        sp.fit(X)
    assert adjusted_rand_score(y, sp.labels_) == 1
    sp = SpectralClustering(n_clusters=2, gamma=2, random_state=0)
    labels = sp.fit(X).labels_
    assert adjusted_rand_score(y, labels) == 1
    X = check_random_state(10).rand(10, 5) * 10
    kernels_available = kernel_metrics()
    for kern in kernels_available:
        if kern != 'additive_chi2':
            sp = SpectralClustering(n_clusters=2, affinity=kern, random_state=0)
            labels = sp.fit(X).labels_
            assert (X.shape[0],) == labels.shape
    sp = SpectralClustering(n_clusters=2, affinity=lambda x, y: 1, random_state=0)
    labels = sp.fit(X).labels_
    assert (X.shape[0],) == labels.shape

    def histogram(x, y, **kwargs):
        assert kwargs == {}
        return np.minimum(x, y).sum()
    sp = SpectralClustering(n_clusters=2, affinity=histogram, random_state=0)
    labels = sp.fit(X).labels_
    assert (X.shape[0],) == labels.shape