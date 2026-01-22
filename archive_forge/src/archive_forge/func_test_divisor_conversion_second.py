import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_divisor_conversion_second(self):
    assert_(np.dtype('m8[s/100]') == np.dtype('m8[10ms]'))
    assert_(np.dtype('m8[3s/10000]') == np.dtype('m8[300us]'))