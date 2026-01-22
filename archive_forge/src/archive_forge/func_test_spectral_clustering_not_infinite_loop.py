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
def test_spectral_clustering_not_infinite_loop(capsys, monkeypatch):
    """Check that discretize raises LinAlgError when svd never converges.

    Non-regression test for #21380
    """

    def new_svd(*args, **kwargs):
        raise LinAlgError()
    monkeypatch.setattr(np.linalg, 'svd', new_svd)
    vectors = np.ones((10, 4))
    with pytest.raises(LinAlgError, match='SVD did not converge'):
        discretize(vectors)