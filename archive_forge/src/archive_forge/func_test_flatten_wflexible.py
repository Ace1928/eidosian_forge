import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_flatten_wflexible(self):
    w, x, _, _ = self.data
    test = merge_arrays((x, w), flatten=True)
    control = np.array([(1, 1, 2, 3.0), (2, 4, 5, 6.0)], dtype=[('f0', int), ('a', int), ('ba', float), ('bb', int)])
    assert_equal(test, control)
    test = merge_arrays((x, w), flatten=False)
    controldtype = [('f0', int), ('f1', [('a', int), ('b', [('ba', float), ('bb', int), ('bc', [])])])]
    control = np.array([(1.0, (1, (2, 3.0, ()))), (2, (4, (5, 6.0, ())))], dtype=controldtype)
    assert_equal(test, control)