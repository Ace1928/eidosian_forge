import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_dropna_preserve_name(self, datetime_series):
    datetime_series[:5] = np.nan
    result = datetime_series.dropna()
    assert result.name == datetime_series.name
    name = datetime_series.name
    ts = datetime_series.copy()
    return_value = ts.dropna(inplace=True)
    assert return_value is None
    assert ts.name == name