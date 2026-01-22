import itertools
import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_,
from pytest import raises as assert_raises
import pytest
from scipy._lib._testutils import check_free_memory
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate._fitpack_py import (splrep, splev, bisplrep, bisplev,
from scipy.interpolate.dfitpack import regrid_smth
from scipy.interpolate._fitpack2 import dfitpack_int
def test_bisplev_integer_overflow():
    np.random.seed(1)
    x = np.linspace(0, 1, 11)
    y = x
    z = np.random.randn(11, 11).ravel()
    kx = 1
    ky = 1
    nx, tx, ny, ty, c, fp, ier = regrid_smth(x, y, z, None, None, None, None, kx=kx, ky=ky, s=0.0)
    tck = (tx[:nx], ty[:ny], c[:(nx - kx - 1) * (ny - ky - 1)], kx, ky)
    xp = np.zeros([2621440])
    yp = np.zeros([2621440])
    assert_raises((RuntimeError, MemoryError), bisplev, xp, yp, tck)