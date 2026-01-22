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
def test_var_std(self, datetime_frame):
    result = datetime_frame.std(ddof=4)
    expected = datetime_frame.apply(lambda x: x.std(ddof=4))
    tm.assert_almost_equal(result, expected)
    result = datetime_frame.var(ddof=4)
    expected = datetime_frame.apply(lambda x: x.var(ddof=4))
    tm.assert_almost_equal(result, expected)
    arr = np.repeat(np.random.default_rng(2).random((1, 1000)), 1000, 0)
    result = nanops.nanvar(arr, axis=0)
    assert not (result < 0).any()
    with pd.option_context('use_bottleneck', False):
        result = nanops.nanvar(arr, axis=0)
        assert not (result < 0).any()