from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_detrend_mean_ValueError(self):
    for signal, kwargs in [(5.5, {'axis': 0}), (self.sig_slope, {'axis': 1}), (self.sig_slope[np.newaxis], {'axis': 2})]:
        with pytest.raises(ValueError):
            mlab.detrend_mean(signal, **kwargs)