import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
@pytest.mark.skipif(not IS_64BIT, reason='Needs 64bit platform')
def test_integer_split_2D_rows_greater_max_int32(self):
    a = np.broadcast_to([0], (1 << 32, 2))
    res = array_split(a, 4)
    chunk = np.broadcast_to([0], (1 << 30, 2))
    tgt = [chunk] * 4
    for i in range(len(tgt)):
        assert_equal(res[i].shape, tgt[i].shape)