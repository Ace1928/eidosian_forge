from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_to_timedelta_bad_value_coerce(self):
    tm.assert_index_equal(TimedeltaIndex([pd.NaT, pd.NaT]), to_timedelta(['foo', 'bar'], errors='coerce'))
    tm.assert_index_equal(TimedeltaIndex(['1 day', pd.NaT, '1 min']), to_timedelta(['1 day', 'bar', '1 min'], errors='coerce'))