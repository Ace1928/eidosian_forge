from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_to_timedelta_units(self):
    result = TimedeltaIndex([np.timedelta64(0, 'ns'), np.timedelta64(10, 's').astype('m8[ns]')])
    expected = to_timedelta([0, 10], unit='s')
    tm.assert_index_equal(result, expected)