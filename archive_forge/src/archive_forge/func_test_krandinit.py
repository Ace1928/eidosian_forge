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
@pytest.mark.skipif(sys.platform == 'win32', reason='Fails with MemoryError in Wine.')
def test_krandinit(self, xp):
    data = xp.asarray(TESTDATA_2D)
    datas = [xp.reshape(data, (200, 2)), xp.reshape(data, (20, 20))[:10, :]]
    k = int(1000000.0)
    for data in datas:
        rng = np.random.default_rng(1234)
        init = _krandinit(data, k, rng, xp)
        orig_cov = cov(data.T)
        init_cov = cov(init.T)
        xp_assert_close(orig_cov, init_cov, atol=0.01)