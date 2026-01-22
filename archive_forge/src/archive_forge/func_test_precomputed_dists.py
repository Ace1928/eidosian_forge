import warnings
import numpy as np
import pytest
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.cluster._optics import _extend_region, _extract_xi_labels
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', [None] + CSR_CONTAINERS)
def test_precomputed_dists(global_dtype, csr_container):
    redX = X[::2].astype(global_dtype, copy=False)
    dists = pairwise_distances(redX, metric='euclidean')
    dists = csr_container(dists) if csr_container is not None else dists
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', EfficiencyWarning)
        clust1 = OPTICS(min_samples=10, algorithm='brute', metric='precomputed').fit(dists)
    clust2 = OPTICS(min_samples=10, algorithm='brute', metric='euclidean').fit(redX)
    assert_allclose(clust1.reachability_, clust2.reachability_)
    assert_array_equal(clust1.labels_, clust2.labels_)