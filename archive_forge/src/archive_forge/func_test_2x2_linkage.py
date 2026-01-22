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
def test_2x2_linkage(xp):
    Z1 = linkage(xp.asarray([1]), method='single', metric='euclidean')
    Z2 = linkage(xp.asarray([[0, 1], [0, 0]]), method='single', metric='euclidean')
    xp_assert_close(Z1, Z2, rtol=1e-15)