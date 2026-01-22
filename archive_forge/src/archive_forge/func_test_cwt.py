import numpy as np
from numpy.testing import (assert_equal,
import pytest
import scipy.signal._wavelets as wavelets
def test_cwt(self):
    with pytest.deprecated_call():
        widths = [1.0]

        def delta_wavelet(s, t):
            return np.array([1])
        len_data = 100
        test_data = np.sin(np.pi * np.arange(0, len_data) / 10.0)
        cwt_dat = wavelets.cwt(test_data, delta_wavelet, widths)
        assert_(cwt_dat.shape == (len(widths), len_data))
        assert_array_almost_equal(test_data, cwt_dat.flatten())
        widths = [1, 3, 4, 5, 10]
        cwt_dat = wavelets.cwt(test_data, wavelets.ricker, widths)
        assert_(cwt_dat.shape == (len(widths), len_data))
        widths = [len_data * 10]

        def flat_wavelet(l, w):
            return np.full(w, 1 / w)
        cwt_dat = wavelets.cwt(test_data, flat_wavelet, widths)
        assert_array_almost_equal(cwt_dat, np.mean(test_data))