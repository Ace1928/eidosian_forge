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
def test_pyfunc_metric():
    X = np.random.random((10, 3))
    euclidean = DistanceMetric.get_metric('euclidean')
    pyfunc = DistanceMetric.get_metric('pyfunc', func=dist_func, p=2)
    euclidean_pkl = pickle.loads(pickle.dumps(euclidean))
    pyfunc_pkl = pickle.loads(pickle.dumps(pyfunc))
    D1 = euclidean.pairwise(X)
    D2 = pyfunc.pairwise(X)
    D1_pkl = euclidean_pkl.pairwise(X)
    D2_pkl = pyfunc_pkl.pairwise(X)
    assert_allclose(D1, D2)
    assert_allclose(D1_pkl, D2_pkl)