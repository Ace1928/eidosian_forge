import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
@pytest.mark.parametrize('box', [None, 'series', 'index'])
def test_constructor_datetime64arr_ok(self, box):
    data = date_range('2017', periods=4, freq='ME')
    if box is None:
        data = data._values
    elif box == 'series':
        data = Series(data)
    result = PeriodIndex(data, freq='D')
    expected = PeriodIndex(['2017-01-31', '2017-02-28', '2017-03-31', '2017-04-30'], freq='D')
    tm.assert_index_equal(result, expected)