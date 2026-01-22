from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_agg_consistency_int_str_column_mix():
    df = DataFrame(np.random.default_rng(2).standard_normal((1000, 2)), index=date_range('1/1/2012', freq='s', periods=1000), columns=[1, 'a'])
    r = df.resample('3min')
    msg = "Column\\(s\\) \\[2, 'b'\\] do not exist"
    with pytest.raises(KeyError, match=msg):
        r.agg({2: 'mean', 'b': 'sum'})