import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_divisor_conversion_minute(self):
    assert_(np.dtype('m8[m/30]') == np.dtype('m8[2s]'))
    assert_(np.dtype('m8[3m/300]') == np.dtype('m8[600ms]'))