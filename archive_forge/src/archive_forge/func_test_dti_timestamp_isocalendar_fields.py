from pandas import (
import pandas._testing as tm
def test_dti_timestamp_isocalendar_fields():
    idx = date_range('2020-01-01', periods=10)
    expected = tuple(idx.isocalendar().iloc[-1].to_list())
    result = idx[-1].isocalendar()
    assert result == expected