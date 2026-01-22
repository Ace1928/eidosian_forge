import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_multi_index_right(self, trades, quotes):
    trades = trades.set_index('time')
    quotes = quotes.set_index(['time', 'bid'])
    with pytest.raises(MergeError, match='right can only have one index'):
        merge_asof(trades, quotes, left_index=True, right_index=True)