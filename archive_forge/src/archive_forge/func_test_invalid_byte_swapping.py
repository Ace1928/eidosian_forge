import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, IS_PYPY
def test_invalid_byte_swapping(self):
    dt = np.dtype('=i8').newbyteorder()
    x = np.arange(5, dtype=dt)
    with pytest.raises(BufferError):
        np.from_dlpack(x)