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
def check_is_valid_im_various_size(self, nrow, ncol, valid, xp):
    R = xp.asarray([[0, 1, 3.0, 2, 5], [3, 2, 4.0, 3, 3]], dtype=xp.float64)
    R = R[:nrow, :ncol]
    assert_(is_valid_im(R) == valid)
    if not valid:
        assert_raises(ValueError, is_valid_im, R, throw=True)