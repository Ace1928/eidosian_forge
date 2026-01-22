from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
@pytest.mark.parametrize('readonly', [True, False])
def test_to_timedelta_readonly(self, readonly):
    arr = np.array([], dtype=object)
    if readonly:
        arr.setflags(write=False)
    result = to_timedelta(arr)
    expected = to_timedelta([])
    tm.assert_index_equal(result, expected)