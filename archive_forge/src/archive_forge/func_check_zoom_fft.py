import pytest
from numpy.testing import assert_allclose
from scipy.fft import fft
from scipy.signal import (czt, zoom_fft, czt_points, CZT, ZoomFFT)
import numpy as np
def check_zoom_fft(x):
    y = fft(x)
    y1 = zoom_fft(x, [0, 2 - 2.0 / len(y)], endpoint=True)
    assert_allclose(y1, y, rtol=1e-11, atol=1e-14)
    y1 = zoom_fft(x, [0, 2])
    assert_allclose(y1, y, rtol=1e-11, atol=1e-14)
    y1 = zoom_fft(x, 2 - 2.0 / len(y), endpoint=True)
    assert_allclose(y1, y, rtol=1e-11, atol=1e-14)
    y1 = zoom_fft(x, 2)
    assert_allclose(y1, y, rtol=1e-11, atol=1e-14)
    over = 10
    yover = fft(x, over * len(x))
    y2 = zoom_fft(x, [0, 2 - 2.0 / len(yover)], m=len(yover), endpoint=True)
    assert_allclose(y2, yover, rtol=1e-12, atol=1e-10)
    y2 = zoom_fft(x, [0, 2], m=len(yover))
    assert_allclose(y2, yover, rtol=1e-12, atol=1e-10)
    w = np.linspace(0, 2 - 2.0 / len(x), len(x))
    f1, f2 = (w[3], w[6])
    y3 = zoom_fft(x, [f1, f2], m=3 * over + 1, endpoint=True)
    idx3 = slice(3 * over, 6 * over + 1)
    assert_allclose(y3, yover[idx3], rtol=1e-13)