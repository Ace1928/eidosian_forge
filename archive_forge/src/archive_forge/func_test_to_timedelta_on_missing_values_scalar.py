from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
@pytest.mark.parametrize('val', [np.nan, pd.NaT, pd.NA])
def test_to_timedelta_on_missing_values_scalar(self, val):
    actual = to_timedelta(val)
    assert actual._value == np.timedelta64('NaT').astype('int64')