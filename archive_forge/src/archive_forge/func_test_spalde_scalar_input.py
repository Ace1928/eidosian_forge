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
def test_spalde_scalar_input():
    x = np.linspace(0, 10)
    y = x ** 3
    tck = splrep(x, y, k=3, t=[5])
    res = spalde(np.float64(1), tck)
    des = np.array([1.0, 3.0, 6.0, 6.0])
    assert_almost_equal(res, des)