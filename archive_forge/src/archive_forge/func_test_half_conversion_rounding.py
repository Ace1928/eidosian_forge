import platform
import pytest
import numpy as np
from numpy import uint16, float16, float32, float64
from numpy.testing import assert_, assert_equal, _OLD_PROMOTION, IS_WASM
@pytest.mark.parametrize('offset', [None, 'up', 'down'])
@pytest.mark.parametrize('shift', [None, 'up', 'down'])
@pytest.mark.parametrize('float_t', [np.float32, np.float64])
@np._no_nep50_warning()
def test_half_conversion_rounding(self, float_t, shift, offset):
    max_pattern = np.float16(np.finfo(np.float16).max).view(np.uint16)
    f16s_patterns = np.arange(0, max_pattern + 1, dtype=np.uint16)
    f16s_float = f16s_patterns.view(np.float16).astype(float_t)
    if shift == 'up':
        f16s_float = 0.5 * (f16s_float[:-1] + f16s_float[1:])[1:]
    elif shift == 'down':
        f16s_float = 0.5 * (f16s_float[:-1] + f16s_float[1:])[:-1]
    else:
        f16s_float = f16s_float[1:-1]
    if offset == 'up':
        f16s_float = np.nextafter(f16s_float, float_t(np.inf))
    elif offset == 'down':
        f16s_float = np.nextafter(f16s_float, float_t(-np.inf))
    res_patterns = f16s_float.astype(np.float16).view(np.uint16)
    cmp_patterns = f16s_patterns[1:-1].copy()
    if shift == 'down' and offset != 'up':
        shift_pattern = -1
    elif shift == 'up' and offset != 'down':
        shift_pattern = 1
    else:
        shift_pattern = 0
    if offset is None:
        cmp_patterns[0::2].view(np.int16)[...] += shift_pattern
    else:
        cmp_patterns.view(np.int16)[...] += shift_pattern
    assert_equal(res_patterns, cmp_patterns)