from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_psd_window_hanning_detrend_linear(self):
    if self.NFFT_density is None:
        return
    ydata = np.arange(self.NFFT_density)
    ycontrol = np.zeros(self.NFFT_density)
    ydata1 = ydata + 5
    ydata2 = ydata + 3.3
    ycontrol1 = ycontrol
    ycontrol2 = ycontrol
    windowVals = mlab.window_hanning(np.ones_like(ycontrol1))
    ycontrol1 = ycontrol1 * windowVals
    ycontrol2 = mlab.window_hanning(ycontrol2)
    ydata = np.vstack([ydata1, ydata2])
    ycontrol = np.vstack([ycontrol1, ycontrol2])
    ydata = np.tile(ydata, (20, 1))
    ycontrol = np.tile(ycontrol, (20, 1))
    ydatab = ydata.T.flatten()
    ydataf = ydata.flatten()
    ycontrol = ycontrol.flatten()
    spec_g, fsp_g = mlab.psd(x=ydataf, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides, detrend=mlab.detrend_linear, window=mlab.window_hanning)
    spec_b, fsp_b = mlab.psd(x=ydatab, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides, detrend=mlab.detrend_linear, window=mlab.window_hanning)
    spec_c, fsp_c = mlab.psd(x=ycontrol, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides, window=mlab.window_none)
    spec_c *= len(ycontrol1) / (windowVals ** 2).sum()
    assert_array_equal(fsp_g, fsp_c)
    assert_array_equal(fsp_b, fsp_c)
    assert_allclose(spec_g, spec_c, atol=1e-08)
    with pytest.raises(AssertionError):
        assert_allclose(spec_b, spec_c, atol=1e-08)