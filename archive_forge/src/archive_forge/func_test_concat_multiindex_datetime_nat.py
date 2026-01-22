import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_multiindex_datetime_nat():
    left = DataFrame({'a': 1}, index=MultiIndex.from_tuples([(1, pd.NaT)]))
    right = DataFrame({'b': 2}, index=MultiIndex.from_tuples([(1, pd.NaT), (2, pd.NaT)]))
    result = concat([left, right], axis='columns')
    expected = DataFrame({'a': [1.0, np.nan], 'b': 2}, MultiIndex.from_tuples([(1, pd.NaT), (2, pd.NaT)]))
    tm.assert_frame_equal(result, expected)