from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_detrend_ValueError(self):
    for signal, kwargs in [(self.sig_slope[np.newaxis], {'key': 'spam'}), (self.sig_slope[np.newaxis], {'key': 5}), (5.5, {'axis': 0}), (self.sig_slope, {'axis': 1}), (self.sig_slope[np.newaxis], {'axis': 2})]:
        with pytest.raises(ValueError):
            mlab.detrend(signal, **kwargs)