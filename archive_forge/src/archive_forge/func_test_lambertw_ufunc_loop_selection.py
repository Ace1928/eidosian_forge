import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_almost_equal
from scipy.special import lambertw
from numpy import nan, inf, pi, e, isnan, log, r_, array, complex128
from scipy.special._testutils import FuncData
def test_lambertw_ufunc_loop_selection():
    dt = np.dtype(np.complex128)
    assert_equal(lambertw(0, 0, 0).dtype, dt)
    assert_equal(lambertw([0], 0, 0).dtype, dt)
    assert_equal(lambertw(0, [0], 0).dtype, dt)
    assert_equal(lambertw(0, 0, [0]).dtype, dt)
    assert_equal(lambertw([0], [0], [0]).dtype, dt)