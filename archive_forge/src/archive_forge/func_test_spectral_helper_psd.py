from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
@pytest.mark.parametrize('mode, case', [('psd', 'density'), ('magnitude', 'specgram'), ('magnitude', 'spectrum')])
def test_spectral_helper_psd(self, mode, case):
    freqs = getattr(self, f'freqs_{case}')
    spec, fsp, t = mlab._spectral_helper(x=self.y, y=self.y, NFFT=getattr(self, f'NFFT_{case}'), Fs=self.Fs, noverlap=getattr(self, f'nover_{case}'), pad_to=getattr(self, f'pad_to_{case}'), sides=self.sides, mode=mode)
    assert_allclose(fsp, freqs, atol=1e-06)
    assert_allclose(t, getattr(self, f't_{case}'), atol=1e-06)
    assert spec.shape[0] == freqs.shape[0]
    assert spec.shape[1] == getattr(self, f't_{case}').shape[0]