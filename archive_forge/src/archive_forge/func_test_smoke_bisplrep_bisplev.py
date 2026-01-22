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
def test_smoke_bisplrep_bisplev(self):
    xb, xe = (0, 2.0 * np.pi)
    yb, ye = (0, 2.0 * np.pi)
    kx, ky = (3, 3)
    Nx, Ny = (20, 20)

    def f2(x, y):
        return np.sin(x + y)
    x = np.linspace(xb, xe, Nx + 1)
    y = np.linspace(yb, ye, Ny + 1)
    xy = makepairs(x, y)
    tck = bisplrep(xy[0], xy[1], f2(xy[0], xy[1]), s=0, kx=kx, ky=ky)
    tt = [tck[0][kx:-kx], tck[1][ky:-ky]]
    t2 = makepairs(tt[0], tt[1])
    v1 = bisplev(tt[0], tt[1], tck)
    v2 = f2(t2[0], t2[1])
    v2.shape = (len(tt[0]), len(tt[1]))
    assert norm2(np.ravel(v1 - v2)) < 0.01