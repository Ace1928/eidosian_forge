from datetime import (
import itertools
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_iat_dont_wrap_object_datetimelike():
    dti = date_range('2016-01-01', periods=3)
    tdi = dti - dti
    ser = Series(dti.to_pydatetime(), dtype=object)
    ser2 = Series(tdi.to_pytimedelta(), dtype=object)
    df = DataFrame({'A': ser, 'B': ser2})
    assert (df.dtypes == object).all()
    for result in [df.at[0, 'A'], df.iat[0, 0], df.loc[0, 'A'], df.iloc[0, 0]]:
        assert result is ser[0]
        assert isinstance(result, datetime)
        assert not isinstance(result, Timestamp)
    for result in [df.at[1, 'B'], df.iat[1, 1], df.loc[1, 'B'], df.iloc[1, 1]]:
        assert result is ser2[1]
        assert isinstance(result, timedelta)
        assert not isinstance(result, Timedelta)