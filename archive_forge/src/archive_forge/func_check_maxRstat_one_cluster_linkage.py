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
def check_maxRstat_one_cluster_linkage(self, i, xp):
    Z = xp.asarray([[0, 1, 0.3, 4]], dtype=xp.float64)
    R = xp.asarray([[0, 0, 0, 0.3]], dtype=xp.float64)
    MD = maxRstat(Z, R, 1)
    expectedMD = calculate_maximum_inconsistencies(Z, R, 1, xp)
    xp_assert_close(MD, expectedMD, atol=1e-15)