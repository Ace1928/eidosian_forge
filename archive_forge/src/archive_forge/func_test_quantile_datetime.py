import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_quantile_datetime(self, unit):
    dti = pd.to_datetime(['2010', '2011']).as_unit(unit)
    df = DataFrame({'a': dti, 'b': [0, 5]})
    result = df.quantile(0.5, numeric_only=True)
    expected = Series([2.5], index=['b'], name=0.5)
    tm.assert_series_equal(result, expected)
    result = df.quantile(0.5, numeric_only=False)
    expected = Series([Timestamp('2010-07-02 12:00:00'), 2.5], index=['a', 'b'], name=0.5)
    tm.assert_series_equal(result, expected)
    result = df.quantile([0.5], numeric_only=False)
    expected = DataFrame({'a': Timestamp('2010-07-02 12:00:00').as_unit(unit), 'b': 2.5}, index=[0.5])
    tm.assert_frame_equal(result, expected)
    df['c'] = pd.to_datetime(['2011', '2012']).as_unit(unit)
    result = df[['a', 'c']].quantile(0.5, axis=1, numeric_only=False)
    expected = Series([Timestamp('2010-07-02 12:00:00'), Timestamp('2011-07-02 12:00:00')], index=[0, 1], name=0.5, dtype=f'M8[{unit}]')
    tm.assert_series_equal(result, expected)
    result = df[['a', 'c']].quantile([0.5], axis=1, numeric_only=False)
    expected = DataFrame([[Timestamp('2010-07-02 12:00:00'), Timestamp('2011-07-02 12:00:00')]], index=[0.5], columns=[0, 1], dtype=f'M8[{unit}]')
    tm.assert_frame_equal(result, expected)
    result = df[['a', 'c']].quantile(0.5, numeric_only=True)
    expected = Series([], index=[], dtype=np.float64, name=0.5)
    tm.assert_series_equal(result, expected)
    result = df[['a', 'c']].quantile([0.5], numeric_only=True)
    expected = DataFrame(index=[0.5], columns=[])
    tm.assert_frame_equal(result, expected)