import platform
import pytest
import numpy as np
from numpy import uint16, float16, float32, float64
from numpy.testing import assert_, assert_equal, _OLD_PROMOTION, IS_WASM
def test_spacing_nextafter(self):
    """Test np.spacing and np.nextafter"""
    a = np.arange(31744, dtype=uint16)
    hinf = np.array((np.inf,), dtype=float16)
    hnan = np.array((np.nan,), dtype=float16)
    a_f16 = a.view(dtype=float16)
    assert_equal(np.spacing(a_f16[:-1]), a_f16[1:] - a_f16[:-1])
    assert_equal(np.nextafter(a_f16[:-1], hinf), a_f16[1:])
    assert_equal(np.nextafter(a_f16[0], -hinf), -a_f16[1])
    assert_equal(np.nextafter(a_f16[1:], -hinf), a_f16[:-1])
    assert_equal(np.nextafter(hinf, a_f16), a_f16[-1])
    assert_equal(np.nextafter(-hinf, a_f16), -a_f16[-1])
    assert_equal(np.nextafter(hinf, hinf), hinf)
    assert_equal(np.nextafter(hinf, -hinf), a_f16[-1])
    assert_equal(np.nextafter(-hinf, hinf), -a_f16[-1])
    assert_equal(np.nextafter(-hinf, -hinf), -hinf)
    assert_equal(np.nextafter(a_f16, hnan), hnan[0])
    assert_equal(np.nextafter(hnan, a_f16), hnan[0])
    assert_equal(np.nextafter(hnan, hnan), hnan)
    assert_equal(np.nextafter(hinf, hnan), hnan)
    assert_equal(np.nextafter(hnan, hinf), hnan)
    a |= 32768
    assert_equal(np.spacing(a_f16[0]), np.spacing(a_f16[1]))
    assert_equal(np.spacing(a_f16[1:]), a_f16[:-1] - a_f16[1:])
    assert_equal(np.nextafter(a_f16[0], hinf), -a_f16[1])
    assert_equal(np.nextafter(a_f16[1:], hinf), a_f16[:-1])
    assert_equal(np.nextafter(a_f16[:-1], -hinf), a_f16[1:])
    assert_equal(np.nextafter(hinf, a_f16), -a_f16[-1])
    assert_equal(np.nextafter(-hinf, a_f16), a_f16[-1])
    assert_equal(np.nextafter(a_f16, hnan), hnan[0])
    assert_equal(np.nextafter(hnan, a_f16), hnan[0])