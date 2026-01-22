from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_partial_slicing_dataframe(self):
    formats = ['%Y', '%Y-%m', '%Y-%m-%d', '%Y-%m-%d %H', '%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S']
    resolutions = ['year', 'month', 'day', 'hour', 'minute', 'second']
    for rnum, resolution in enumerate(resolutions[2:], 2):
        unit = Timedelta('1 ' + resolution)
        middate = datetime(2012, 1, 1, 0, 0, 0)
        index = DatetimeIndex([middate - unit, middate, middate + unit])
        values = [1, 2, 3]
        df = DataFrame({'a': values}, index, dtype=np.int64)
        assert df.index.resolution == resolution
        for timestamp, expected in zip(index, values):
            ts_string = timestamp.strftime(formats[rnum])
            result = df['a'][ts_string]
            assert isinstance(result, np.int64)
            assert result == expected
            msg = f"^'{ts_string}'$"
            with pytest.raises(KeyError, match=msg):
                df[ts_string]
        for fmt in formats[:rnum]:
            for element, theslice in [[0, slice(None, 1)], [1, slice(1, None)]]:
                ts_string = index[element].strftime(fmt)
                result = df['a'][ts_string]
                expected = df['a'][theslice]
                tm.assert_series_equal(result, expected)
                with pytest.raises(KeyError, match=ts_string):
                    df[ts_string]
        for fmt in formats[rnum + 1:]:
            ts_string = index[1].strftime(fmt)
            result = df['a'][ts_string]
            assert isinstance(result, np.int64)
            assert result == 2
            msg = f"^'{ts_string}'$"
            with pytest.raises(KeyError, match=msg):
                df[ts_string]
        for fmt, res in list(zip(formats, resolutions))[rnum + 1:]:
            ts = index[1] + Timedelta('1 ' + res)
            ts_string = ts.strftime(fmt)
            msg = f"^'{ts_string}'$"
            with pytest.raises(KeyError, match=msg):
                df['a'][ts_string]
            with pytest.raises(KeyError, match=msg):
                df[ts_string]