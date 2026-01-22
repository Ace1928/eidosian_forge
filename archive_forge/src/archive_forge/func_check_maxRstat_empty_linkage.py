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
def check_maxRstat_empty_linkage(self, i, xp):
    Z = xp.zeros((0, 4), dtype=xp.float64)
    R = xp.zeros((0, 4), dtype=xp.float64)
    assert_raises(ValueError, maxRstat, Z, R, i)