import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_different_field_order(self):
    a = np.zeros(3, dtype=[('a', 'i4'), ('b', 'f4'), ('c', 'u1')])
    b = np.ones(3, dtype=[('c', 'u1'), ('b', 'f4'), ('a', 'i4')])
    j = join_by(['c', 'b'], a, b, jointype='inner', usemask=False)
    assert_equal(j.dtype.names, ['b', 'c', 'a1', 'a2'])