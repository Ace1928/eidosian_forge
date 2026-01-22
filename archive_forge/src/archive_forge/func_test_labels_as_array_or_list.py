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
def test_labels_as_array_or_list(self, xp):
    Z = linkage(xp.asarray(hierarchy_test_data.ytdist), 'single')
    labels = xp.asarray([1, 3, 2, 6, 4, 5])
    result1 = dendrogram(Z, labels=labels, no_plot=True)
    result2 = dendrogram(Z, labels=list(labels), no_plot=True)
    assert result1 == result2