import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
def test_max_nan_bug():
    raw = ',Date,app,File\n-04-23,2013-04-23 00:00:00,,log080001.log\n-05-06,2013-05-06 00:00:00,,log.log\n-05-07,2013-05-07 00:00:00,OE,xlsx'
    with tm.assert_produces_warning(UserWarning, match='Could not infer format'):
        df = pd.read_csv(StringIO(raw), parse_dates=[0])
    gb = df.groupby('Date')
    r = gb[['File']].max()
    e = gb['File'].max().to_frame()
    tm.assert_frame_equal(r, e)
    assert not r['File'].isna().any()