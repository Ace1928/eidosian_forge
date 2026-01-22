import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_divisor_conversion_fs(self):
    assert_(np.dtype('M8[fs/100]') == np.dtype('M8[10as]'))
    assert_raises(ValueError, lambda: np.dtype('M8[3fs/10000]'))