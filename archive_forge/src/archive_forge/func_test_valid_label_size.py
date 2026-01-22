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
def test_valid_label_size(self, xp):
    link = xp.asarray([[0, 1, 1.0, 4], [2, 3, 1.0, 5], [4, 5, 2.0, 6]])
    plt.figure()
    with pytest.raises(ValueError) as exc_info:
        dendrogram(link, labels=list(range(100)))
    assert 'Dimensions of Z and labels must be consistent.' in str(exc_info.value)
    with pytest.raises(ValueError, match='Dimensions of Z and labels must be consistent.'):
        dendrogram(link, labels=[])
    plt.close()