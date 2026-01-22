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
def test_warn_if_metric_bool_data_no_bool():
    pairwise_metric = 'rogerstanimoto'
    X = np.random.randint(2, size=(5, 2), dtype=np.int32)
    msg = f'Data will be converted to boolean for metric {pairwise_metric}'
    with pytest.warns(DataConversionWarning, match=msg) as warn_record:
        OPTICS(metric=pairwise_metric).fit(X)
        assert len(warn_record) == 1