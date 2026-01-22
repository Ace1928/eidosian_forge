import re
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_consistency_with_tz_aware_scalar(self):
    df = Series([Timestamp('2016-03-30 14:35:25', tz='Europe/Brussels')]).to_frame()
    df = pd.concat([df, df]).reset_index(drop=True)
    expected = Timestamp('2016-03-30 14:35:25+0200', tz='Europe/Brussels')
    result = df[0][0]
    assert result == expected
    result = df.iloc[0, 0]
    assert result == expected
    result = df.loc[0, 0]
    assert result == expected
    result = df.iat[0, 0]
    assert result == expected
    result = df.at[0, 0]
    assert result == expected
    result = df[0].loc[0]
    assert result == expected
    result = df[0].at[0]
    assert result == expected