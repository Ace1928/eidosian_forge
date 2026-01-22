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
def test_kink(self):
    spl2 = insert(0.5, self.spl, m=2)
    splder(spl2, 2)
    assert_raises(ValueError, splder, spl2, 3)
    spl2 = insert(0.5, self.spl, m=3)
    splder(spl2, 1)
    assert_raises(ValueError, splder, spl2, 2)
    spl2 = insert(0.5, self.spl, m=4)
    assert_raises(ValueError, splder, spl2, 1)