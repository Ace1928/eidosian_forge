import pytest
from numpy.testing import assert_allclose
from scipy.fft import fft
from scipy.signal import (czt, zoom_fft, czt_points, CZT, ZoomFFT)
import numpy as np
@pytest.mark.slow
def test_czt_vs_fft():
    np.random.seed(123)
    random_lengths = np.random.exponential(100000, size=10).astype('int')
    for n in random_lengths:
        a = np.random.randn(n)
        assert_allclose(czt(a), fft(a), rtol=1e-11)