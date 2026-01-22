import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_find_duplicates_ignoremask(self):
    ndtype = [('a', int)]
    a = ma.array([1, 1, 1, 2, 2, 3, 3], mask=[0, 0, 1, 0, 0, 0, 1]).view(ndtype)
    test = find_duplicates(a, ignoremask=True, return_index=True)
    control = [0, 1, 3, 4]
    assert_equal(sorted(test[-1]), control)
    assert_equal(test[0], a[test[-1]])
    test = find_duplicates(a, ignoremask=False, return_index=True)
    control = [0, 1, 2, 3, 4, 6]
    assert_equal(sorted(test[-1]), control)
    assert_equal(test[0], a[test[-1]])