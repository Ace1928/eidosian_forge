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
def test_mlab_linkage_conversion_multiple_rows(self, xp):
    Zm = xp.asarray([[3, 6, 138], [4, 5, 219], [1, 8, 255], [2, 9, 268], [7, 10, 295]])
    Z = xp.asarray([[2.0, 5.0, 138.0, 2.0], [3.0, 4.0, 219.0, 2.0], [0.0, 7.0, 255.0, 3.0], [1.0, 8.0, 268.0, 4.0], [6.0, 9.0, 295.0, 6.0]], dtype=xp.float64)
    xp_assert_close(from_mlab_linkage(Zm), Z, rtol=1e-15)
    xp_assert_close(to_mlab_linkage(Z), xp.asarray(Zm, dtype=xp.float64), rtol=1e-15)