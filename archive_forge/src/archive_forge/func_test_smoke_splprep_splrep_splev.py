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
@pytest.mark.parametrize('N', [20, 50])
@pytest.mark.parametrize('k', [1, 2, 3, 4, 5])
def test_smoke_splprep_splrep_splev(self, N, k):
    a, b, dx = (0, 2.0 * np.pi, 0.2 * np.pi)
    x = np.linspace(a, b, N + 1)
    v = np.sin(x)
    tckp, u = splprep([x, v], s=0, per=0, k=k, nest=-1)
    uv = splev(dx, tckp)
    err1 = abs(uv[1] - np.sin(uv[0]))
    assert err1 < 0.01
    tck = splrep(x, v, s=0, per=0, k=k)
    err2 = abs(splev(uv[0], tck) - np.sin(uv[0]))
    assert err2 < 0.01
    if k == 3:
        tckp, u = splprep([x, v], s=0, per=0, k=k, nest=-1)
        for d in range(1, k + 1):
            uv = splev(dx, tckp, d)