import re
import sys
from io import StringIO
import numpy as np
import pytest
from scipy import sparse as sp
from sklearn.base import clone
from sklearn.cluster import KMeans, MiniBatchKMeans, k_means, kmeans_plusplus
from sklearn.cluster._k_means_common import (
from sklearn.cluster._kmeans import _labels_inertia, _mini_batch_step
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils._testing import (
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import CSR_CONTAINERS, threadpool_limits
@pytest.mark.parametrize('input_data', [X] + X_as_any_csr, ids=data_containers_ids)
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_scaled_weights(Estimator, input_data, global_random_seed):
    sample_weight = np.random.RandomState(global_random_seed).uniform(size=n_samples)
    km = Estimator(n_clusters=n_clusters, random_state=global_random_seed, n_init=1)
    km_orig = clone(km).fit(input_data, sample_weight=sample_weight)
    km_scaled = clone(km).fit(input_data, sample_weight=0.5 * sample_weight)
    assert_array_equal(km_orig.labels_, km_scaled.labels_)
    assert_allclose(km_orig.cluster_centers_, km_scaled.cluster_centers_)