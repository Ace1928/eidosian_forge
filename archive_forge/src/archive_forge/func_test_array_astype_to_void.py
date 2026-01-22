import sys
import numpy as np
from numpy.core._rational_tests import rational
import pytest
from numpy.testing import (
@pytest.mark.parametrize('dt', ['d', 'f', 'S13', 'U32'])
def test_array_astype_to_void(dt):
    dt = np.dtype(dt)
    arr = np.array([], dtype=dt)
    assert arr.astype('V').dtype.itemsize == dt.itemsize