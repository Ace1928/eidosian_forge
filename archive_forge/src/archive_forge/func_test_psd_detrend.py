from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
@pytest.mark.parametrize('make_data, detrend', [(np.zeros, mlab.detrend_mean), (np.zeros, 'mean'), (np.arange, mlab.detrend_linear), (np.arange, 'linear')])
def test_psd_detrend(self, make_data, detrend):
    if self.NFFT_density is None:
        return
    ydata = make_data(self.NFFT_density)
    ydata1 = ydata + 5
    ydata2 = ydata + 3.3
    ydata = np.vstack([ydata1, ydata2])
    ydata = np.tile(ydata, (20, 1))
    ydatab = ydata.T.flatten()
    ydata = ydata.flatten()
    ycontrol = np.zeros_like(ydata)
    spec_g, fsp_g = mlab.psd(x=ydata, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides, detrend=detrend)
    spec_b, fsp_b = mlab.psd(x=ydatab, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides, detrend=detrend)
    spec_c, fsp_c = mlab.psd(x=ycontrol, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides)
    assert_array_equal(fsp_g, fsp_c)
    assert_array_equal(fsp_b, fsp_c)
    assert_allclose(spec_g, spec_c, atol=1e-08)
    with pytest.raises(AssertionError):
        assert_allclose(spec_b, spec_c, atol=1e-08)