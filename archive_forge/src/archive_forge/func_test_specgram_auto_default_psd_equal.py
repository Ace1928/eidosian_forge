from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
@pytest.mark.parametrize('mode', ['default', 'psd'])
def test_specgram_auto_default_psd_equal(self, mode):
    """
        Test that mlab.specgram without mode and with mode 'default' and 'psd'
        are all the same.
        """
    speca, freqspeca, ta = mlab.specgram(x=self.y, NFFT=self.NFFT_specgram, Fs=self.Fs, noverlap=self.nover_specgram, pad_to=self.pad_to_specgram, sides=self.sides)
    specb, freqspecb, tb = mlab.specgram(x=self.y, NFFT=self.NFFT_specgram, Fs=self.Fs, noverlap=self.nover_specgram, pad_to=self.pad_to_specgram, sides=self.sides, mode=mode)
    assert_array_equal(speca, specb)
    assert_array_equal(freqspeca, freqspecb)
    assert_array_equal(ta, tb)