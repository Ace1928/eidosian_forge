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
def test_cluster_qr_permutation_invariance():
    random_state = np.random.RandomState(seed=8)
    n_samples, n_components = (100, 5)
    data = random_state.randn(n_samples, n_components)
    perm = random_state.permutation(n_samples)
    assert np.array_equal(cluster_qr(data)[perm], cluster_qr(data[perm]))