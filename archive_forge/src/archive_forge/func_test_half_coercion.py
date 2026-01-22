import platform
import pytest
import numpy as np
from numpy import uint16, float16, float32, float64
from numpy.testing import assert_, assert_equal, _OLD_PROMOTION, IS_WASM
@np._no_nep50_warning()
def test_half_coercion(self, weak_promotion):
    """Test that half gets coerced properly with the other types"""
    a16 = np.array((1,), dtype=float16)
    a32 = np.array((1,), dtype=float32)
    b16 = float16(1)
    b32 = float32(1)
    assert np.power(a16, 2).dtype == float16
    assert np.power(a16, 2.0).dtype == float16
    assert np.power(a16, b16).dtype == float16
    expected_dt = float32 if weak_promotion else float16
    assert np.power(a16, b32).dtype == expected_dt
    assert np.power(a16, a16).dtype == float16
    assert np.power(a16, a32).dtype == float32
    expected_dt = float16 if weak_promotion else float64
    assert np.power(b16, 2).dtype == expected_dt
    assert np.power(b16, 2.0).dtype == expected_dt
    assert np.power(b16, b16).dtype, float16
    assert np.power(b16, b32).dtype, float32
    assert np.power(b16, a16).dtype, float16
    assert np.power(b16, a32).dtype, float32
    assert np.power(a32, a16).dtype == float32
    assert np.power(a32, b16).dtype == float32
    expected_dt = float32 if weak_promotion else float16
    assert np.power(b32, a16).dtype == expected_dt
    assert np.power(b32, b16).dtype == float32