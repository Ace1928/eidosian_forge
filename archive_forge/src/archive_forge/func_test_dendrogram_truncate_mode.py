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
@pytest.mark.skipif(not have_matplotlib, reason='no matplotlib')
def test_dendrogram_truncate_mode(self, xp):
    Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
    R = dendrogram(Z, 2, 'lastp', show_contracted=True)
    plt.close()
    R['dcoord'] = np.asarray(R['dcoord'])
    assert_equal(R, {'color_list': ['C0'], 'dcoord': [[0.0, 295.0, 295.0, 0.0]], 'icoord': [[5.0, 5.0, 15.0, 15.0]], 'ivl': ['(2)', '(4)'], 'leaves': [6, 9], 'leaves_color_list': ['C0', 'C0']})
    R = dendrogram(Z, 2, 'mtica', show_contracted=True)
    plt.close()
    R['dcoord'] = np.asarray(R['dcoord'])
    assert_equal(R, {'color_list': ['C1', 'C0', 'C0', 'C0'], 'dcoord': [[0.0, 138.0, 138.0, 0.0], [0.0, 255.0, 255.0, 0.0], [0.0, 268.0, 268.0, 255.0], [138.0, 295.0, 295.0, 268.0]], 'icoord': [[5.0, 5.0, 15.0, 15.0], [35.0, 35.0, 45.0, 45.0], [25.0, 25.0, 40.0, 40.0], [10.0, 10.0, 32.5, 32.5]], 'ivl': ['2', '5', '1', '0', '(2)'], 'leaves': [2, 5, 1, 0, 7], 'leaves_color_list': ['C1', 'C1', 'C0', 'C0', 'C0']})