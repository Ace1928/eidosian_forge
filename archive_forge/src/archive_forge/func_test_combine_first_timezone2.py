from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_combine_first_timezone2(self, unit):
    dts1 = pd.date_range('2015-01-01', '2015-01-05', tz='UTC', unit=unit)
    df1 = DataFrame({'DATE': dts1})
    dts2 = pd.date_range('2015-01-03', '2015-01-05', tz='UTC', unit=unit)
    df2 = DataFrame({'DATE': dts2})
    res = df1.combine_first(df2)
    tm.assert_frame_equal(res, df1)
    assert res['DATE'].dtype == f'datetime64[{unit}, UTC]'