from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
@pytest.mark.parametrize('timezone, year, month, day, hour', [['America/Chicago', 2013, 11, 3, 1], ['America/Santiago', 2021, 4, 3, 23]])
def test_reindex_timestamp_with_fold(self, timezone, year, month, day, hour):
    test_timezone = gettz(timezone)
    transition_1 = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=0, fold=0, tzinfo=test_timezone)
    transition_2 = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=0, fold=1, tzinfo=test_timezone)
    df = DataFrame({'index': [transition_1, transition_2], 'vals': ['a', 'b']}).set_index('index').reindex(['1', '2'])
    exp = DataFrame({'index': ['1', '2'], 'vals': [np.nan, np.nan]}).set_index('index')
    exp = exp.astype(df.vals.dtype)
    tm.assert_frame_equal(df, exp)