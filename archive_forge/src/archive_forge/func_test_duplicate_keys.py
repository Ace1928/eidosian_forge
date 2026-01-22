import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_duplicate_keys(self):
    a = np.zeros(3, dtype=[('a', 'i4'), ('b', 'f4'), ('c', 'u1')])
    b = np.ones(3, dtype=[('c', 'u1'), ('b', 'f4'), ('a', 'i4')])
    assert_raises(ValueError, join_by, ['a', 'b', 'b'], a, b)