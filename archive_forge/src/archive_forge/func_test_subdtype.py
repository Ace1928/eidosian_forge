import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_subdtype(self):
    z = np.array([('A', 1), ('B', 2)], dtype=[('A', '|S3'), ('B', float, (1,))])
    zz = np.array([('a', [10.0], 100.0), ('b', [20.0], 200.0), ('c', [30.0], 300.0)], dtype=[('A', '|S3'), ('B', float, (1,)), ('C', float)])
    res = stack_arrays((z, zz))
    expected = ma.array(data=[(b'A', [1.0], 0), (b'B', [2.0], 0), (b'a', [10.0], 100.0), (b'b', [20.0], 200.0), (b'c', [30.0], 300.0)], mask=[(False, [False], True), (False, [False], True), (False, [False], False), (False, [False], False), (False, [False], False)], dtype=zz.dtype)
    assert_equal(res.dtype, expected.dtype)
    assert_equal(res, expected)
    assert_equal(res.mask, expected.mask)