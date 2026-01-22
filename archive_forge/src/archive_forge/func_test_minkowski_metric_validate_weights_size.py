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
def test_minkowski_metric_validate_weights_size():
    w2 = rng.random_sample(d + 1)
    dm = DistanceMetric.get_metric('minkowski', p=3, w=w2)
    msg = f'MinkowskiDistance: the size of w must match the number of features \\({X64.shape[1]}\\). Currently len\\(w\\)={w2.shape[0]}.'
    with pytest.raises(ValueError, match=msg):
        dm.pairwise(X64, Y64)