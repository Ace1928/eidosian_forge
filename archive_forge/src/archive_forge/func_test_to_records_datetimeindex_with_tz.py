from collections import abc
import email
from email.parser import Parser
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', ['UTC', 'GMT', 'US/Eastern'])
def test_to_records_datetimeindex_with_tz(self, tz):
    dr = date_range('2016-01-01', periods=10, freq='s', tz=tz)
    df = DataFrame({'datetime': dr}, index=dr)
    expected = df.to_records()
    result = df.tz_convert('UTC').to_records()
    tm.assert_numpy_array_equal(result, expected)