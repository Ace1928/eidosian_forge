import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_has_nested_dtype(self):
    """Test has_nested_dtype"""
    ndtype = np.dtype(float)
    assert_equal(has_nested_fields(ndtype), False)
    ndtype = np.dtype([('A', '|S3'), ('B', float)])
    assert_equal(has_nested_fields(ndtype), False)
    ndtype = np.dtype([('A', int), ('B', [('BA', float), ('BB', '|S1')])])
    assert_equal(has_nested_fields(ndtype), True)