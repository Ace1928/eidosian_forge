import platform
import pytest
import numpy as np
from numpy import uint16, float16, float32, float64
from numpy.testing import assert_, assert_equal, _OLD_PROMOTION, IS_WASM
def test_half_correctness(self):
    """Take every finite float16, and check the casting functions with
           a manual conversion."""
    a_bits = self.finite_f16.view(dtype=uint16)
    a_sgn = (-1.0) ** ((a_bits & 32768) >> 15)
    a_exp = np.array((a_bits & 31744) >> 10, dtype=np.int32) - 15
    a_man = (a_bits & 1023) * 2.0 ** (-10)
    a_man[a_exp != -15] += 1
    a_exp[a_exp == -15] = -14
    a_manual = a_sgn * a_man * 2.0 ** a_exp
    a32_fail = np.nonzero(self.finite_f32 != a_manual)[0]
    if len(a32_fail) != 0:
        bad_index = a32_fail[0]
        assert_equal(self.finite_f32, a_manual, 'First non-equal is half value 0x%x -> %g != %g' % (a_bits[bad_index], self.finite_f32[bad_index], a_manual[bad_index]))
    a64_fail = np.nonzero(self.finite_f64 != a_manual)[0]
    if len(a64_fail) != 0:
        bad_index = a64_fail[0]
        assert_equal(self.finite_f64, a_manual, 'First non-equal is half value 0x%x -> %g != %g' % (a_bits[bad_index], self.finite_f64[bad_index], a_manual[bad_index]))