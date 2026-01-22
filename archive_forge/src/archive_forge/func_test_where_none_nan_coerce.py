from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_none_nan_coerce():
    expected = DataFrame({'A': [Timestamp('20130101'), pd.NaT, Timestamp('20130103')], 'B': [1, 2, np.nan]})
    result = expected.where(expected.notnull(), None)
    tm.assert_frame_equal(result, expected)