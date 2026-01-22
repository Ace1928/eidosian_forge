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
def check_maxRstat_difrow_linkage(self, i, xp):
    Z = xp.asarray([[0, 1, 0.3, 4]], dtype=xp.float64)
    R = np.random.rand(2, 4)
    R = xp.asarray(R)
    assert_raises(ValueError, maxRstat, Z, R, i)