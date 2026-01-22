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
def test_isna_numpy_nat(self):
    arr = np.array([NaT, np.datetime64('NaT'), np.timedelta64('NaT'), np.datetime64('NaT', 's')])
    result = isna(arr)
    expected = np.array([True] * 4)
    tm.assert_numpy_array_equal(result, expected)