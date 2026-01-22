from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('utc', [True, False])
def test_series_groupby_value_counts_with_grouper(utc):
    df = DataFrame({'Timestamp': [1565083561, 1565083561 + 86400, 1565083561 + 86500, 1565083561 + 86400 * 2, 1565083561 + 86400 * 3, 1565083561 + 86500 * 3, 1565083561 + 86400 * 4], 'Food': ['apple', 'apple', 'banana', 'banana', 'orange', 'orange', 'pear']}).drop([3])
    df['Datetime'] = to_datetime(df['Timestamp'], utc=utc, unit='s')
    dfg = df.groupby(Grouper(freq='1D', key='Datetime'))
    result = dfg['Food'].value_counts().sort_index()
    expected = dfg['Food'].apply(Series.value_counts).sort_index()
    expected.index.names = result.index.names
    expected = expected.rename('count')
    tm.assert_series_equal(result, expected)