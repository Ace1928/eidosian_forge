import warnings
import sys
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.cluster.vq import (kmeans, kmeans2, py_vq, vq, whiten,
from scipy.cluster import _vq
from scipy.conftest import (
from scipy.sparse._sputils import matrix
from scipy._lib._array_api import (
@skip_if_array_api_gpu
@array_api_compatible
def test_kmeans2_kpp_low_dim(self, xp):
    prev_res = xp.asarray([[-1.95266667, 0.898], [-3.153375, 3.3945]], dtype=xp.float64)
    np.random.seed(42)
    res, _ = kmeans2(xp.asarray(TESTDATA_2D), xp.asarray(2), minit='++')
    xp_assert_close(res, prev_res)