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
def check_linkage_tdist(self, method, xp):
    Z = linkage(xp.asarray(hierarchy_test_data.ytdist), method)
    expectedZ = getattr(hierarchy_test_data, 'linkage_ytdist_' + method)
    xp_assert_close(Z, xp.asarray(expectedZ), atol=1e-10)