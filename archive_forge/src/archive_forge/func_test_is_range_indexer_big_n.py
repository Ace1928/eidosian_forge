import numpy as np
import pytest
from pandas._libs import (
from pandas.compat import IS64
from pandas import Index
import pandas._testing as tm
@pytest.mark.skipif(not IS64, reason="2**31 is too big for Py_ssize_t on 32-bit. It doesn't matter though since you cannot create an array that long on 32-bit")
@pytest.mark.parametrize('dtype', ['int64', 'int32'])
def test_is_range_indexer_big_n(self, dtype):
    left = np.arange(0, 100, dtype=dtype)
    assert not lib.is_range_indexer(left, 2 ** 31)