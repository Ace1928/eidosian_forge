from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_csd(self):
    freqs = self.freqs_density
    spec, fsp = mlab.csd(x=self.y, y=self.y + 1, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=self.nover_density, pad_to=self.pad_to_density, sides=self.sides)
    assert_allclose(fsp, freqs, atol=1e-06)
    assert spec.shape == freqs.shape