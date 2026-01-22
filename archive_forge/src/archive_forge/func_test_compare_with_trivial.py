import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_, assert_warns
import pytest
from pytest import raises as assert_raises
import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import (
from scipy.spatial.distance import pdist
from scipy.cluster._hierarchy import Heap
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close
from . import hierarchy_test_data
@skip_if_array_api_gpu
@array_api_compatible
def test_compare_with_trivial(self, xp):
    rng = np.random.RandomState(0)
    n = 20
    X = rng.rand(n, 2)
    d = pdist(X)
    for method, code in _LINKAGE_METHODS.items():
        Z_trivial = _hierarchy.linkage(d, n, code)
        Z = linkage(xp.asarray(d), method)
        xp_assert_close(Z, xp.asarray(Z_trivial), rtol=1e-14, atol=1e-15)