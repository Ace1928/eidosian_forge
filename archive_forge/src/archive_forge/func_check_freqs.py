from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def check_freqs(self, vals, targfreqs, resfreqs, fstims):
    assert resfreqs.argmin() == 0
    assert resfreqs.argmax() == len(resfreqs) - 1
    assert_allclose(resfreqs, targfreqs, atol=1e-06)
    for fstim in fstims:
        i = np.abs(resfreqs - fstim).argmin()
        assert vals[i] > vals[i + 2]
        assert vals[i] > vals[i - 2]