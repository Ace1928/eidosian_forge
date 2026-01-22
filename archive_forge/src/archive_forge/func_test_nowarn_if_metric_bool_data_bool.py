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
def test_nowarn_if_metric_bool_data_bool():
    pairwise_metric = 'rogerstanimoto'
    X = np.random.randint(2, size=(5, 2), dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DataConversionWarning)
        OPTICS(metric=pairwise_metric).fit(X)