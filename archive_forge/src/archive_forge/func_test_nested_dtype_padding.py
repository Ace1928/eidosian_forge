import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_nested_dtype_padding(self):
    """ test that trailing padding is preserved """
    dt = np.dtype([('a', np.uint8), ('b', np.uint8), ('c', np.uint8)])
    dt_padded_end = dt[['a', 'b']]
    assert dt_padded_end.itemsize == dt.itemsize
    dt_outer = np.dtype([('inner', dt_padded_end)])
    data = np.zeros(3, dt_outer).view(np.recarray)
    assert_equal(data['inner'].dtype, dt_padded_end)
    data0 = data[0]
    assert_equal(data0['inner'].dtype, dt_padded_end)