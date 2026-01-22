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
@skip_if_array_api_backend('numpy.array_api')
def test_cut_tree(xp):
    np.random.seed(23)
    nobs = 50
    X = np.random.randn(nobs, 4)
    X = xp.asarray(X)
    Z = scipy.cluster.hierarchy.ward(X)
    cutree = cut_tree(Z)
    xp_assert_close(cutree[:, 0], xp.arange(nobs), rtol=1e-15, check_dtype=False)
    xp_assert_close(cutree[:, -1], xp.zeros(nobs), rtol=1e-15, check_dtype=False)
    assert_equal(np.asarray(cutree).max(0), np.arange(nobs - 1, -1, -1))
    xp_assert_close(cutree[:, [-5]], cut_tree(Z, n_clusters=5), rtol=1e-15)
    xp_assert_close(cutree[:, [-5, -10]], cut_tree(Z, n_clusters=[5, 10]), rtol=1e-15)
    xp_assert_close(cutree[:, [-10, -5]], cut_tree(Z, n_clusters=[10, 5]), rtol=1e-15)
    nodes = _order_cluster_tree(Z)
    heights = xp.asarray([node.dist for node in nodes])
    xp_assert_close(cutree[:, np.searchsorted(heights, [5])], cut_tree(Z, height=5), rtol=1e-15)
    xp_assert_close(cutree[:, np.searchsorted(heights, [5, 10])], cut_tree(Z, height=[5, 10]), rtol=1e-15)
    xp_assert_close(cutree[:, np.searchsorted(heights, [10, 5])], cut_tree(Z, height=[10, 5]), rtol=1e-15)