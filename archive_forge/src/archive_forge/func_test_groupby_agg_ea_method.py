from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
def test_groupby_agg_ea_method(monkeypatch):

    def DecimalArray__my_sum(self):
        return np.sum(np.array(self))
    monkeypatch.setattr(DecimalArray, 'my_sum', DecimalArray__my_sum, raising=False)
    data = make_data()[:5]
    df = pd.DataFrame({'id': [0, 0, 0, 1, 1], 'decimals': DecimalArray(data)})
    expected = pd.Series(to_decimal([data[0] + data[1] + data[2], data[3] + data[4]]))
    result = df.groupby('id')['decimals'].agg(lambda x: x.values.my_sum())
    tm.assert_series_equal(result, expected, check_names=False)
    s = pd.Series(DecimalArray(data))
    grouper = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    result = s.groupby(grouper).agg(lambda x: x.values.my_sum())
    tm.assert_series_equal(result, expected, check_names=False)