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
@array_api_compatible
def test_num_obs_linkage_2x4(self, xp):
    Z = xp.asarray([[0, 1, 3.0, 2], [3, 2, 4.0, 3]], dtype=xp.float64)
    assert_equal(num_obs_linkage(Z), 3)