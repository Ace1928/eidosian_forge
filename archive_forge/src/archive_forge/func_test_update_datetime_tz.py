import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_update_datetime_tz(self):
    result = DataFrame([pd.Timestamp('2019', tz='UTC')])
    with tm.assert_produces_warning(None):
        result.update(result)
    expected = DataFrame([pd.Timestamp('2019', tz='UTC')])
    tm.assert_frame_equal(result, expected)