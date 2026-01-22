from __future__ import annotations
from datetime import timedelta
import operator
import numpy as np
import pytest
from pandas._libs.tslibs import tz_compare
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_add_mismatched_reso_doesnt_downcast(self):
    td = pd.Timedelta(microseconds=1)
    dti = pd.date_range('2016-01-01', periods=3) - td
    dta = dti._data.as_unit('us')
    res = dta + td.as_unit('us')
    assert res.unit == 'us'