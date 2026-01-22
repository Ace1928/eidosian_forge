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
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_predict_sample_weight_deprecation_warning(Estimator):
    X = np.random.rand(100, 2)
    sample_weight = np.random.uniform(size=100)
    kmeans = Estimator()
    kmeans.fit(X, sample_weight=sample_weight)
    warn_msg = "'sample_weight' was deprecated in version 1.3 and will be removed in 1.5."
    with pytest.warns(FutureWarning, match=warn_msg):
        kmeans.predict(X, sample_weight=sample_weight)