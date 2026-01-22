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
def test_dendrogram_single_linkage_tdist(self, xp):
    Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
    R = dendrogram(Z, no_plot=True)
    leaves = R['leaves']
    assert_equal(leaves, [2, 5, 1, 0, 3, 4])