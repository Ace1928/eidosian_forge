import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
import scipy.interpolate.interpnd as interpnd
import scipy.spatial._qhull as qhull
import pickle
def test_square_rescale(self):
    points = np.array([(0, 0), (0, 100), (10, 100), (10, 0)], dtype=np.float64)
    values = np.array([1.0, 2.0, -3.0, 5.0], dtype=np.float64)
    xx, yy = np.broadcast_arrays(np.linspace(0, 10, 14)[:, None], np.linspace(0, 100, 14)[None, :])
    xx = xx.ravel()
    yy = yy.ravel()
    xi = np.array([xx, yy]).T.copy()
    zi = interpnd.LinearNDInterpolator(points, values)(xi)
    zi_rescaled = interpnd.LinearNDInterpolator(points, values, rescale=True)(xi)
    assert_almost_equal(zi, zi_rescaled)