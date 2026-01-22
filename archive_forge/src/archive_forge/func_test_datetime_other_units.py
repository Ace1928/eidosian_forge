from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._config import config as cf
from pandas._libs import missing as libmissing
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_datetime_other_units(self):
    idx = DatetimeIndex(['2011-01-01', 'NaT', '2011-01-02'])
    exp = np.array([False, True, False])
    tm.assert_numpy_array_equal(isna(idx), exp)
    tm.assert_numpy_array_equal(notna(idx), ~exp)
    tm.assert_numpy_array_equal(isna(idx.values), exp)
    tm.assert_numpy_array_equal(notna(idx.values), ~exp)