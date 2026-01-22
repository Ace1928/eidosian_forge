import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.parametrize('freq', ['D', 'M', 'Y'])
def test_pickle_round_trip(self, freq):
    idx = PeriodIndex(['2016-05-16', 'NaT', NaT, np.nan], freq=freq)
    result = tm.round_trip_pickle(idx)
    tm.assert_index_equal(result, idx)