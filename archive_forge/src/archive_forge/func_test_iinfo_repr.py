import warnings
import numpy as np
import pytest
from numpy.core import finfo, iinfo
from numpy import half, single, double, longdouble
from numpy.testing import assert_equal, assert_, assert_raises
from numpy.core.getlimits import _discovered_machar, _float_ma
def test_iinfo_repr(self):
    expected = 'iinfo(min=-32768, max=32767, dtype=int16)'
    assert_equal(repr(np.iinfo(np.int16)), expected)