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
def test_kmeans_lost_cluster(self, xp):
    data = xp.asarray(TESTDATA_2D)
    initk = xp.asarray([[-1.8127404, -0.67128041], [2.04621601, 0.07401111], [-2.31149087, -0.05160469]])
    kmeans(data, initk)
    with suppress_warnings() as sup:
        sup.filter(UserWarning, 'One of the clusters is empty. Re-run kmeans with a different initialization')
        kmeans2(data, initk, missing='warn')
    assert_raises(ClusterError, kmeans2, data, initk, missing='raise')