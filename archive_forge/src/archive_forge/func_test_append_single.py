import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_append_single(self):
    _, x, _, _ = self.data
    test = append_fields(x, 'A', data=[10, 20, 30])
    control = ma.array([(1, 10), (2, 20), (-1, 30)], mask=[(0, 0), (0, 0), (1, 0)], dtype=[('f0', int), ('A', int)])
    assert_equal(test, control)