import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_singlerecord(self):
    _, x, y, z = self.data
    test = merge_arrays((x[0], y[0], z[0]), usemask=False)
    control = np.array([(1, 10, ('A', 1))], dtype=[('f0', int), ('f1', int), ('f2', [('A', '|S3'), ('B', float)])])
    assert_equal(test, control)