import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_leftouter_join(self):
    a, b = (self.a, self.b)
    test = join_by(('a', 'b'), a, b, 'leftouter')
    control = ma.array([(0, 50, 100, -1), (1, 51, 101, -1), (2, 52, 102, -1), (3, 53, 103, -1), (4, 54, 104, -1), (5, 55, 105, -1), (6, 56, 106, -1), (7, 57, 107, -1), (8, 58, 108, -1), (9, 59, 109, -1)], mask=[(0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1)], dtype=[('a', int), ('b', int), ('c', int), ('d', int)])
    assert_equal(test, control)