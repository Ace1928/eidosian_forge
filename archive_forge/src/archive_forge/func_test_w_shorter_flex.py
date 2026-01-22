import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_w_shorter_flex(self):
    z = self.data[-1]
    merge_arrays((z, np.array([10, 20, 30]).view([('C', int)])))
    np.array([('A', 1.0, 10), ('B', 2.0, 20), ('-1', -1, 20)], dtype=[('A', '|S3'), ('B', float), ('C', int)])