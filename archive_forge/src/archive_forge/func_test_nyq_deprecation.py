import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test_nyq_deprecation(self):
    with pytest.warns(DeprecationWarning, match="Keyword argument 'nyq' is deprecated in "):
        firls(1, (0, 1), (0, 0), nyq=10)