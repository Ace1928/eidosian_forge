import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_valid_allow_exact_matches(self, trades, quotes):
    msg = 'allow_exact_matches must be boolean, passed foo'
    with pytest.raises(MergeError, match=msg):
        merge_asof(trades, quotes, on='time', by='ticker', allow_exact_matches='foo')