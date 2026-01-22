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
def test_large_features(self, xp):
    d = 300
    n = 100
    m1 = np.random.randn(d)
    m2 = np.random.randn(d)
    x = 10000 * np.random.randn(n, d) - 20000 * m1
    y = 10000 * np.random.randn(n, d) + 20000 * m2
    data = np.empty((x.shape[0] + y.shape[0], d), np.float64)
    data[:x.shape[0]] = x
    data[x.shape[0]:] = y
    kmeans(xp.asarray(data), xp.asarray(2))