from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
@pytest.mark.parametrize('mode, conv', [('magnitude', np.abs), ('angle', np.angle), ('phase', lambda x: np.unwrap(np.angle(x), axis=0))])
def test_specgram_complex_equivalent(self, mode, conv):
    specc, freqspecc, tc = mlab.specgram(x=self.y, NFFT=self.NFFT_specgram, Fs=self.Fs, noverlap=self.nover_specgram, pad_to=self.pad_to_specgram, sides=self.sides, mode='complex')
    specm, freqspecm, tm = mlab.specgram(x=self.y, NFFT=self.NFFT_specgram, Fs=self.Fs, noverlap=self.nover_specgram, pad_to=self.pad_to_specgram, sides=self.sides, mode=mode)
    assert_array_equal(freqspecc, freqspecm)
    assert_array_equal(tc, tm)
    assert_allclose(conv(specc), specm, atol=1e-06)