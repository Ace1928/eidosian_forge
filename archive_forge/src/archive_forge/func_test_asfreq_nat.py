import re
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_asfreq_nat(self):
    idx = PeriodIndex(['2011-01', '2011-02', 'NaT', '2011-04'], freq='M')
    result = idx.asfreq(freq='Q')
    expected = PeriodIndex(['2011Q1', '2011Q1', 'NaT', '2011Q2'], freq='Q')
    tm.assert_index_equal(result, expected)