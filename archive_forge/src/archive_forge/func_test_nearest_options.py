import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.interpolate import (griddata, NearestNDInterpolator,
def test_nearest_options(self):
    npts, nd = (4, 3)
    x = np.arange(npts * nd).reshape((npts, nd))
    y = np.arange(npts)
    nndi = NearestNDInterpolator(x, y)
    opts = {'balanced_tree': False, 'compact_nodes': False}
    nndi_o = NearestNDInterpolator(x, y, tree_options=opts)
    assert_allclose(nndi(x), nndi_o(x), atol=1e-14)