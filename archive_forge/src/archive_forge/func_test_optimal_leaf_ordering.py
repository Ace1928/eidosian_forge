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
def test_optimal_leaf_ordering(self, xp):
    Z = linkage(xp.asarray(hierarchy_test_data.ytdist), optimal_ordering=True)
    expectedZ = getattr(hierarchy_test_data, 'linkage_ytdist_single_olo')
    xp_assert_close(Z, xp.asarray(expectedZ), atol=1e-10)