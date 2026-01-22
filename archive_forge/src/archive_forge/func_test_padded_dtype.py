import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_padded_dtype(self):
    dt = np.dtype('i1,f4', align=True)
    dt.names = ('k', 'v')
    assert_(len(dt.descr), 3)
    a = np.array([(1, 3), (3, 2)], dt)
    b = np.array([(1, 1), (2, 2)], dt)
    res = join_by('k', a, b)
    expected_dtype = np.dtype([('k', 'i1'), ('v1', 'f4'), ('v2', 'f4')])
    assert_equal(res.dtype, expected_dtype)