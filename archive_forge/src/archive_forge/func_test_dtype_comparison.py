import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_dtype_comparison(self):
    assert_(not np.dtype('M8[us]') == np.dtype('M8[ms]'))
    assert_(np.dtype('M8[us]') != np.dtype('M8[ms]'))
    assert_(np.dtype('M8[2D]') != np.dtype('M8[D]'))
    assert_(np.dtype('M8[D]') != np.dtype('M8[2D]'))