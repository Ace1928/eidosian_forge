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
def test_kmeans2_rank1_2(self, xp):
    data = xp.asarray(TESTDATA_2D)
    data1 = data[:, 0]
    kmeans2(data1, xp.asarray(2), iter=1)