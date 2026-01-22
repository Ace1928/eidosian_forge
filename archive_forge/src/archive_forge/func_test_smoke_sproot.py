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
def test_smoke_sproot(self):
    a, b = (0.1, 15)
    x = np.linspace(a, b, 20)
    v = np.sin(x)
    for k in [1, 2, 4, 5]:
        tck = splrep(x, v, s=0, per=0, k=k, xe=b)
        with assert_raises(ValueError):
            sproot(tck)
    k = 3
    tck = splrep(x, v, s=0, k=3)
    roots = sproot(tck)
    assert_allclose(splev(roots, tck), 0, atol=1e-10, rtol=1e-10)
    assert_allclose(roots, np.pi * np.array([1, 2, 3, 4]), rtol=0.001)