from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_psd_window_flattop(self):
    a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
    fac = np.linspace(-np.pi, np.pi, self.NFFT_density_real)
    win = np.zeros(self.NFFT_density_real)
    for k in range(len(a)):
        win += a[k] * np.cos(k * fac)
    spec, fsp = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides, window=win, scale_by_freq=False)
    spec_a, fsp_a = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides, window=win)
    assert_allclose(spec * win.sum() ** 2, spec_a * self.Fs * (win ** 2).sum(), atol=1e-08)