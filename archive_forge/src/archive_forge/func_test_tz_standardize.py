import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_tz_standardize(self):
    tz = pytz.timezone('US/Eastern')
    dr = date_range('2013-01-01', periods=3, tz='US/Eastern')
    dtype = DatetimeTZDtype('ns', dr.tz)
    assert dtype.tz == tz
    dtype = DatetimeTZDtype('ns', dr[0].tz)
    assert dtype.tz == tz