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
def test_kmeans2_init(self, xp):
    np.random.seed(12345)
    data = xp.asarray(TESTDATA_2D)
    k = xp.asarray(3)
    kmeans2(data, k, minit='points')
    kmeans2(data[:, :1], k, minit='points')
    kmeans2(data, k, minit='++')
    kmeans2(data[:, :1], k, minit='++')
    with suppress_warnings() as sup:
        sup.filter(message='One of the clusters is empty. Re-run.')
        kmeans2(data, k, minit='random')
        kmeans2(data[:, :1], k, minit='random')