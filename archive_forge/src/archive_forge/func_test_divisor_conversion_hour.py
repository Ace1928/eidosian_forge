import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_divisor_conversion_hour(self):
    assert_(np.dtype('m8[h/30]') == np.dtype('m8[2m]'))
    assert_(np.dtype('m8[3h/300]') == np.dtype('m8[36s]'))