from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_to_timedelta_same_np_timedelta64(self):
    result = to_timedelta(np.array([np.timedelta64(1, 's')]))
    expected = pd.Index(np.array([np.timedelta64(1, 's')]))
    tm.assert_index_equal(result, expected)