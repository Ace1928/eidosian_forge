from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_to_timedelta_ignore_strings_unit(self):
    arr = np.array([1, 2, 'error'], dtype=object)
    msg = "errors='ignore' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = to_timedelta(arr, unit='ns', errors='ignore')
    tm.assert_numpy_array_equal(result, arr)