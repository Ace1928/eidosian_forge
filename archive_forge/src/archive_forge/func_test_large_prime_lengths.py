import pytest
from numpy.testing import assert_allclose
from scipy.fft import fft
from scipy.signal import (czt, zoom_fft, czt_points, CZT, ZoomFFT)
import numpy as np
def test_large_prime_lengths():
    np.random.seed(0)
    for N in (101, 1009, 10007):
        x = np.random.rand(N)
        y = fft(x)
        y1 = czt(x)
        assert_allclose(y, y1, rtol=1e-12)