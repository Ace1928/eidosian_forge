from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_psd_windowarray_scale_by_freq(self):
    win = mlab.window_hanning(np.ones(self.NFFT_density_real))
    spec, fsp = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=self.nover_density, pad_to=self.pad_to_density, sides=self.sides, window=mlab.window_hanning)
    spec_s, fsp_s = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=self.nover_density, pad_to=self.pad_to_density, sides=self.sides, window=mlab.window_hanning, scale_by_freq=True)
    spec_n, fsp_n = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=self.nover_density, pad_to=self.pad_to_density, sides=self.sides, window=mlab.window_hanning, scale_by_freq=False)
    assert_array_equal(fsp, fsp_s)
    assert_array_equal(fsp, fsp_n)
    assert_array_equal(spec, spec_s)
    assert_allclose(spec_s * (win ** 2).sum(), spec_n / self.Fs * win.sum() ** 2, atol=1e-08)