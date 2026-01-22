import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, IS_PYPY
def test_strides_not_multiple_of_itemsize(self):
    dt = np.dtype([('int', np.int32), ('char', np.int8)])
    y = np.zeros((5,), dtype=dt)
    z = y['int']
    with pytest.raises(BufferError):
        np.from_dlpack(z)