import numpy as np
import pytest
from pandas._libs import (
from pandas.compat import IS64
from pandas import Index
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['int64', 'int32'])
def test_is_range_indexer(self, dtype):
    left = np.arange(0, 100, dtype=dtype)
    assert lib.is_range_indexer(left, 100)