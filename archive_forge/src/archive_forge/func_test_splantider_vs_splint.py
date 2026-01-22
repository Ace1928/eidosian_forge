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
def test_splantider_vs_splint(self):
    spl2 = splantider(self.spl)
    xx = np.linspace(0, 1, 20)
    for x1 in xx:
        for x2 in xx:
            y1 = splint(x1, x2, self.spl)
            y2 = splev(x2, spl2) - splev(x1, spl2)
            assert_allclose(y1, y2)