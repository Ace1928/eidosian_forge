import copy
import itertools
import pickle
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.metrics import DistanceMetric
from sklearn.metrics._dist_metrics import (
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, create_memmap_backed_data
from sklearn.utils.fixes import CSR_CONTAINERS, parse_version, sp_version
@pytest.mark.parametrize('metric', BOOL_METRICS)
@pytest.mark.parametrize('X_bool', [X_bool, X_bool_mmap])
def test_pickle_bool_metrics(metric, X_bool):
    dm = DistanceMetric.get_metric(metric)
    D1 = dm.pairwise(X_bool)
    dm2 = pickle.loads(pickle.dumps(dm))
    D2 = dm2.pairwise(X_bool)
    assert_allclose(D1, D2)