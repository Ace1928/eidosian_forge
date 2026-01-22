import numpy as np
import pytest
from scipy.conftest import array_api_compatible
from scipy._lib._array_api import (
import scipy._lib.array_api_compat.array_api_compat.numpy as np_compat
@array_api_compatible
def test_asarray(self, xp):
    x, y = (as_xparray([0, 1, 2], xp=xp), as_xparray(np.arange(3), xp=xp))
    ref = xp.asarray([0, 1, 2])
    xp_assert_equal(x, ref)
    xp_assert_equal(y, ref)