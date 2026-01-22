import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
import scipy.interpolate.interpnd as interpnd
import scipy.spatial._qhull as qhull
import pickle
def test_smoketest_rescale(self):
    x = np.array([(0, 0), (-5, -5), (-5, 5), (5, 5), (2.5, 3)], dtype=np.float64)
    y = np.arange(x.shape[0], dtype=np.float64)
    yi = interpnd.LinearNDInterpolator(x, y, rescale=True)(x)
    assert_almost_equal(y, yi)