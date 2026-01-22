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
def check_inconsistent_tdist(self, depth, xp):
    Z = xp.asarray(hierarchy_test_data.linkage_ytdist_single)
    xp_assert_close(inconsistent(Z, depth), xp.asarray(hierarchy_test_data.inconsistent_ytdist[depth]))