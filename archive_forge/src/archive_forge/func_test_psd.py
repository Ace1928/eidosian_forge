from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_psd(self):
    freqs = self.freqs_density
    spec, fsp = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=self.nover_density, pad_to=self.pad_to_density, sides=self.sides)
    assert spec.shape == freqs.shape
    self.check_freqs(spec, freqs, fsp, self.fstims)