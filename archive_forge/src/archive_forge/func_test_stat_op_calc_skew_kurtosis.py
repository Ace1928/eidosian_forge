from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
def test_stat_op_calc_skew_kurtosis(self, float_frame_with_na):
    sp_stats = pytest.importorskip('scipy.stats')

    def skewness(x):
        if len(x) < 3:
            return np.nan
        return sp_stats.skew(x, bias=False)

    def kurt(x):
        if len(x) < 4:
            return np.nan
        return sp_stats.kurtosis(x, bias=False)
    assert_stat_op_calc('skew', skewness, float_frame_with_na)
    assert_stat_op_calc('kurt', kurt, float_frame_with_na)