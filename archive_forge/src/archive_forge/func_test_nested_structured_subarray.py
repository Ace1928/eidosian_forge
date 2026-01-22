import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_nested_structured_subarray():
    point = np.dtype([('x', float), ('y', float)])
    dt = np.dtype([('code', int), ('points', point, (2,))])
    data = StringIO('100,1,2,3,4\n200,5,6,7,8\n')
    expected = np.array([(100, [(1.0, 2.0), (3.0, 4.0)]), (200, [(5.0, 6.0), (7.0, 8.0)])], dtype=dt)
    assert_array_equal(np.loadtxt(data, dtype=dt, delimiter=','), expected)