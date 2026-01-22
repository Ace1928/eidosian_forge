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
@pytest.mark.parametrize('attr', ['groups', 'ngroups', 'indices'])
def test_api_compat_before_use(attr):
    rng = date_range('1/1/2012', periods=100, freq='s')
    ts = Series(np.arange(len(rng)), index=rng)
    rs = ts.resample('30s')
    getattr(rs, attr)
    rs.mean()
    getattr(rs, attr)