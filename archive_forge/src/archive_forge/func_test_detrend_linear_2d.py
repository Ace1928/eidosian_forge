from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_detrend_linear_2d(self):
    input = np.vstack([self.sig_off, self.sig_slope, self.sig_slope + self.sig_off])
    target = np.vstack([self.sig_zeros, self.sig_zeros, self.sig_zeros])
    self.allclose(mlab.detrend(input.T, key='linear', axis=0), target.T)
    self.allclose(mlab.detrend(input.T, key=mlab.detrend_linear, axis=0), target.T)
    self.allclose(mlab.detrend(input, key='linear', axis=1), target)
    self.allclose(mlab.detrend(input, key=mlab.detrend_linear, axis=1), target)
    with pytest.raises(ValueError):
        mlab.detrend_linear(self.sig_slope[np.newaxis])