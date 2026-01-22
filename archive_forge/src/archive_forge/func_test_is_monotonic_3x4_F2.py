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
def test_is_monotonic_3x4_F2(self, xp):
    Z = xp.asarray([[0, 1, 0.8, 2], [2, 3, 0.4, 2], [4, 5, 0.6, 4]], dtype=xp.float64)
    assert not is_monotonic(Z)