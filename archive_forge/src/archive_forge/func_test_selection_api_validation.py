from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_selection_api_validation():
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq='D')
    rng = np.arange(len(index), dtype=np.int64)
    df = DataFrame({'date': index, 'a': rng}, index=pd.MultiIndex.from_arrays([rng, index], names=['v', 'd']))
    df_exp = DataFrame({'a': rng}, index=index)
    msg = "Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, but got an instance of 'Index'"
    with pytest.raises(TypeError, match=msg):
        df.resample('2D', level='v')
    msg = 'The Grouper cannot specify both a key and a level!'
    with pytest.raises(ValueError, match=msg):
        df.resample('2D', on='date', level='d')
    msg = "unhashable type: 'list'"
    with pytest.raises(TypeError, match=msg):
        df.resample('2D', on=['a', 'date'])
    msg = '\\"Level \\[\'a\', \'date\'\\] not found\\"'
    with pytest.raises(KeyError, match=msg):
        df.resample('2D', level=['a', 'date'])
    msg = 'Upsampling from level= or on= selection is not supported, use \\.set_index\\(\\.\\.\\.\\) to explicitly set index to datetime-like'
    with pytest.raises(ValueError, match=msg):
        df.resample('2D', level='d').asfreq()
    with pytest.raises(ValueError, match=msg):
        df.resample('2D', on='date').asfreq()
    exp = df_exp.resample('2D').sum()
    exp.index.name = 'date'
    result = df.resample('2D', on='date').sum()
    tm.assert_frame_equal(exp, result)
    exp.index.name = 'd'
    with pytest.raises(TypeError, match='datetime64 type does not support sum'):
        df.resample('2D', level='d').sum()
    result = df.resample('2D', level='d').sum(numeric_only=True)
    tm.assert_frame_equal(exp, result)