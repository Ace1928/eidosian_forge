import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
import scipy.interpolate.interpnd as interpnd
import scipy.spatial._qhull as qhull
import pickle
def test_smoketest_alternate(self):
    x = np.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.25, 0.3)], dtype=np.float64)
    y = np.arange(x.shape[0], dtype=np.float64)
    yi = interpnd.LinearNDInterpolator((x[:, 0], x[:, 1]), y)(x[:, 0], x[:, 1])
    assert_almost_equal(y, yi)