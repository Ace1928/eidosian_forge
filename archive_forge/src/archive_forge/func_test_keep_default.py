import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_keep_default(self):
    """Make sure we don't lose an explicit default"""
    converter = StringConverter(None, missing_values='', default=-999)
    converter.upgrade('3.14159265')
    assert_equal(converter.default, -999)
    assert_equal(converter.type, np.dtype(float))
    converter = StringConverter(None, missing_values='', default=0)
    converter.upgrade('3.14159265')
    assert_equal(converter.default, 0)
    assert_equal(converter.type, np.dtype(float))