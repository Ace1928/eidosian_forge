import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_non_sorted(self, trades, quotes):
    trades = trades.sort_values('time', ascending=False)
    quotes = quotes.sort_values('time', ascending=False)
    assert not trades.time.is_monotonic_increasing
    assert not quotes.time.is_monotonic_increasing
    with pytest.raises(ValueError, match='left keys must be sorted'):
        merge_asof(trades, quotes, on='time', by='ticker')
    trades = trades.sort_values('time')
    assert trades.time.is_monotonic_increasing
    assert not quotes.time.is_monotonic_increasing
    with pytest.raises(ValueError, match='right keys must be sorted'):
        merge_asof(trades, quotes, on='time', by='ticker')
    quotes = quotes.sort_values('time')
    assert trades.time.is_monotonic_increasing
    assert quotes.time.is_monotonic_increasing
    merge_asof(trades, quotes, on='time', by='ticker')