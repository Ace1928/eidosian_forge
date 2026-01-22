import pytest
from numpy.testing import assert_allclose
from scipy.fft import fft
from scipy.signal import (czt, zoom_fft, czt_points, CZT, ZoomFFT)
import numpy as np
@pytest.mark.parametrize('impulse', ([0, 0, 1], [0, 0, 1, 0, 0], np.concatenate((np.array([0, 0, 1]), np.zeros(100)))))
@pytest.mark.parametrize('m', (1, 3, 5, 8, 101, 1021))
@pytest.mark.parametrize('a', (1, 2, 0.5, 1.1))
@pytest.mark.parametrize('w', (None, 0.98534 + 0.17055j))
def test_czt_math(impulse, m, w, a):
    assert_allclose(czt(impulse[2:], m=m, w=w, a=a), np.ones(m), rtol=1e-10)
    assert_allclose(czt(impulse[1:], m=m, w=w, a=a), czt_points(m=m, w=w, a=a) ** (-1), rtol=1e-10)
    assert_allclose(czt(impulse, m=m, w=w, a=a), czt_points(m=m, w=w, a=a) ** (-2), rtol=1e-10)