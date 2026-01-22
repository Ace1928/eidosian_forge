import datetime as dt
from functools import partial
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('as_period', [True, False])
def test_agg_tzaware_non_datetime_result(as_period):
    dti = date_range('2012-01-01', periods=4, tz='UTC')
    if as_period:
        dti = dti.tz_localize(None).to_period('D')
    df = DataFrame({'a': [0, 0, 1, 1], 'b': dti})
    gb = df.groupby('a')
    result = gb['b'].agg(lambda x: x.iloc[0])
    expected = Series(dti[::2], name='b')
    expected.index.name = 'a'
    tm.assert_series_equal(result, expected)
    result = gb['b'].agg(lambda x: x.iloc[0].year)
    expected = Series([2012, 2012], name='b')
    expected.index.name = 'a'
    tm.assert_series_equal(result, expected)
    result = gb['b'].agg(lambda x: x.iloc[-1] - x.iloc[0])
    expected = Series([pd.Timedelta(days=1), pd.Timedelta(days=1)], name='b')
    expected.index.name = 'a'
    if as_period:
        expected = Series([pd.offsets.Day(1), pd.offsets.Day(1)], name='b')
        expected.index.name = 'a'
    tm.assert_series_equal(result, expected)