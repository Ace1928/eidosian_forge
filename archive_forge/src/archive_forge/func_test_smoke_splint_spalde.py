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
@pytest.mark.parametrize('per', [0, 1])
def test_smoke_splint_spalde(self, N, per):
    self.check_2(per=per, N=N)