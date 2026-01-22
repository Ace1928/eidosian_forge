import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_append_on_flex(self):
    z = self.data[-1]
    test = append_fields(z, 'C', data=[10, 20, 30])
    control = ma.array([('A', 1.0, 10), ('B', 2.0, 20), (-1, -1.0, 30)], mask=[(0, 0, 0), (0, 0, 0), (1, 1, 0)], dtype=[('A', '|S3'), ('B', float), ('C', int)])
    assert_equal(test, control)