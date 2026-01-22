import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_valid_join_keys(self, trades, quotes):
    msg = 'incompatible merge keys \\[1\\] .* must be the same type'
    with pytest.raises(MergeError, match=msg):
        merge_asof(trades, quotes, left_on='time', right_on='bid', by='ticker')
    with pytest.raises(MergeError, match='can only asof on a key for left'):
        merge_asof(trades, quotes, on=['time', 'ticker'], by='ticker')
    with pytest.raises(MergeError, match='can only asof on a key for left'):
        merge_asof(trades, quotes, by='ticker')