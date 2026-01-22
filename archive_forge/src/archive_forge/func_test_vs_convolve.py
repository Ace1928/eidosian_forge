import numpy as np
from itertools import product
from numpy.testing import assert_equal, assert_allclose
from pytest import raises as assert_raises
import pytest
from scipy.signal import upfirdn, firwin
from scipy.signal._upfirdn import _output_len, _upfirdn_modes
from scipy.signal._upfirdn_apply import _pad_test
@pytest.mark.parametrize('down, want_len', [(2, 5015), (11, 912), (79, 127)])
def test_vs_convolve(self, down, want_len):
    random_state = np.random.RandomState(17)
    try_types = (int, np.float32, np.complex64, float, complex)
    size = 10000
    for dtype in try_types:
        x = random_state.randn(size).astype(dtype)
        if dtype in (np.complex64, np.complex128):
            x += 1j * random_state.randn(size)
        h = firwin(31, 1.0 / down, window='hamming')
        yl = upfirdn_naive(x, h, 1, down)
        y = upfirdn(h, x, up=1, down=down)
        assert y.shape == (want_len,)
        assert yl.shape[0] == y.shape[0]
        assert_allclose(yl, y, atol=1e-07, rtol=1e-07)